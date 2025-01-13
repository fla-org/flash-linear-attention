import os
import torch
from tqdm import tqdm
import wandb
import logging
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import get_scheduler
from torch.amp import GradScaler, autocast
from fla.vision_models.delta_net import DeltaNetVisionConfig, DeltaNetForImageClassification
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 # deafult dtype for FLA

def setup_logging(args):
    log_filename = f'training_{args.model}_vision_{args.dataset}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_filename}")

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Vision Model Training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset name')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--amp_enabled', action='store_true', help='Enable AMP if device supports it')
    parser.add_argument('--b_lr', type=float, default=2e-4, help='Backbone learning rate')
    parser.add_argument('--h_lr', type=float, default=2e-4, help='Head learning rate')
    parser.add_argument('--wd', type=float, default=0., help='Weight decay')
    parser.add_argument('--train_bs', type=int, default=128, help='Training batch size')
    parser.add_argument('--eval_bs', type=int, default=256, help='Eval batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--eval_epoch', type=int, default=1, help='Eval frequency')
    parser.add_argument('--log_step', type=int, default=10, help='Log frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--expand_k', type=float, default=1.0, help='Key expansion ratio')
    parser.add_argument('--expand_v', type=float, default=1.0, help='Value expansion ratio')
    parser.add_argument('--attn_mode', type=str, default='chunk', choices=['chunk', 'fused_recurrent', 'fused_chunk'])
    parser.add_argument('--pool_type', type=str, default='mean', choices=['mean', 'cls'])
    parser.add_argument('--model', type=str, required=True, help='Model type (currently only supports "deltanet")')
    parser.add_argument('--fuse_cross_entropy', action='store_true', help='Fuse cross entropy with logits')
    
    # Learning rate schedule related arguments
    parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup',
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 
                               'constant', 'constant_with_warmup'])
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Ratio of total training steps for warmup')
    # Add hybrid attention related arguments
    parser.add_argument('--use_attn', action='store_true', help='Use hybrid attention in some layers')
    parser.add_argument('--attn_layers', type=str, default='0,1',
                       help='Comma separated list of layer indices to use attention, e.g. "0,1,2"')
    parser.add_argument('--attn_num_heads', type=int, default=16, 
                       help='Number of attention heads for hybrid attention layers')
    parser.add_argument('--attn_num_kv_heads', type=int, default=None,
                       help='Number of key/value heads for hybrid attention layers')
    parser.add_argument('--attn_window_size', type=int, default=None,
                       help='Window size for hybrid attention layers')
    parser.add_argument('--log_memory_epoch', type=int, default=100, help='Log memory usage frequency')
    return parser.parse_args()

def get_data(args):
    """
    Prepare data transforms and loaders.
    Ensures consistent data types with model.
    """
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype),  # Match model dtype
    ])
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=args.num_workers)
    
    return train_loader, test_loader, num_classes

def setup_deterministic_mode(args):
    """Setup deterministic mode for reproducibility"""
    import numpy as np
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_memory_info():
    """
    Get current GPU memory usage information
    Returns a dictionary with:
    - memory_allocated: actively allocated memory
    - memory_reserved: reserved memory in GPU
    - max_memory_allocated: max allocated memory since the beginning
    """
    return {
        'memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
        'memory_reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
        'max_memory_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
    }

def log_gpu_memory(args, epoch):
    """Log GPU memory usage if CUDA is available"""
    if torch.cuda.is_available() and epoch % args.log_memory_epoch == 0:
        memory_info = get_gpu_memory_info()
        logging.info(
            f"GPU Memory Usage (Epoch {epoch}) - "
            f"Allocated: {memory_info['memory_allocated']:.2f}MB, "
            f"Reserved: {memory_info['memory_reserved']:.2f}MB, "
            f"Peak: {memory_info['max_memory_allocated']:.2f}MB"
        )
        if args.wandb:
            wandb.log({
                "gpu_memory/allocated": memory_info['memory_allocated'],
                "gpu_memory/reserved": memory_info['memory_reserved'],
                "gpu_memory/peak": memory_info['max_memory_allocated'],
                "epoch": epoch
            })

def evaluate(model, test_loader, device, args):
    """
    Evaluation loop with proper CUDA timing.
    Uses CUDA events for accurate GPU timing and ensures proper synchronization.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Create CUDA events for timing
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.perf_counter()
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device=device, dtype=dtype)
            targets = targets.to(device)
            
            if args.amp_enabled:
                with autocast():
                    outputs = model(images).logits
                    _, predicted = outputs.max(1)
            else:
                outputs = model(images).logits
                _, predicted = outputs.max(1)
                
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Measure time with proper CUDA synchronization
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        eval_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    else:
        eval_time = time.perf_counter() - start_time
        
    accuracy = 100. * correct / total
    return accuracy, eval_time

def get_model(args, num_classes):
    """
    Initialize model based on configuration.
    Supports both pure DeltaNet and hybrid models.
    """
    if args.model == 'deltanet':
        # Prepare attention config for hybrid model if enabled
        attn_config = None
        if args.use_attn:
            attn_config = {
                'layers': [int(i) for i in args.attn_layers.split(',')],
                'num_heads': args.attn_num_heads,
                'num_kv_heads': args.attn_num_kv_heads,
                'window_size': args.attn_window_size
            }
            # Log hybrid attention configuration
            logging.info("Hybrid Attention Configuration:")
            logging.info(f"- Attention Layers: {attn_config['layers']}")
            logging.info(f"- Number of Heads: {attn_config['num_heads']}")
            logging.info(f"- Number of KV Heads: {attn_config['num_kv_heads']}")
            logging.info(f"- Window Size: {attn_config['window_size']}")

        config = DeltaNetVisionConfig(
            num_hidden_layers=args.num_hidden_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=num_classes,
            expand_k=args.expand_k,
            expand_v=args.expand_v,
            attn_mode=args.attn_mode,
            pool_type=args.pool_type,
            fuse_cross_entropy=args.fuse_cross_entropy,
            attn=attn_config  # Add attention config for hybrid model
        )
        return DeltaNetForImageClassification(config).to(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented yet.")

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args, epoch):
    """
    Training loop for one epoch with proper CUDA timing.
    Uses CUDA events for accurate GPU timing and ensures proper synchronization.
    """
    model.train()
    total_loss = 0
    scaler = GradScaler() if args.amp_enabled else None
    
    # Create CUDA events for timing
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.perf_counter()
    
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device=device, dtype=dtype)
        targets = targets.to(device)
        
        if args.amp_enabled:
            with autocast():
                outputs = model(images).logits
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        optimizer.zero_grad()
        scheduler.step()  # Update learning rate scheduler
        total_loss += loss.item()

        if i % args.log_step == 0:
            lrs = [group['lr'] for group in optimizer.param_groups]
            logging.info(f'Epoch {epoch} Step {i}/{len(train_loader)}: '
                        f'Loss={loss.item():.4f} '
                        f'LR_backbone={lrs[0]:.2e} '
                        f'LR_head={lrs[-1]:.2e}')
            
            if args.wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate/backbone": lrs[0],
                    "learning_rate/head": lrs[-1],
                    "global_step": epoch * len(train_loader) + i
                })
    
    # Measure time with proper CUDA synchronization
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        train_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        train_time = time.perf_counter() - start_time
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss, train_time

def main():
    args = get_args()
    
    # Setup logging first, before any logging calls
    setup_logging(args)
    
    # Then setup deterministic mode
    setup_deterministic_mode(args)
    
    # Log all configuration parameters
    logging.info("=" * 50)
    logging.info("Training Configuration:")
    logging.info("-" * 50)
    for arg, value in sorted(vars(args).items()):
        logging.info(f"{arg}: {value}")
    logging.info("=" * 50)
    
    # Setup wandb after logging is initialized
    if args.wandb:
        project_name = f"{args.model}_vision_classification"
        run_name = f"e{args.epochs}_b_lr{args.b_lr}_h_lr_{args.h_lr}_mode{args.attn_mode}_bs{args.train_bs}_p{args.patch_size}_i{args.image_size}_h{args.num_heads}_{args.dataset}"
        wandb.init(
            project=project_name,
            name=run_name,
            config=args.__dict__
        )
        logging.info(f"Wandb initialized with project: {project_name}, run: {run_name}")
    
    train_loader, test_loader, num_classes = get_data(args)
    
    # Calculate total training steps
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    model = get_model(args, num_classes)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info("=" * 50)
    logging.info("Model Information:")
    logging.info("-" * 50)
    logging.info(f"Model Type: {args.model}")
    logging.info(f"Number of trainable parameters: {trainable_params:,}")
    logging.info(f"Number of layers: {args.num_hidden_layers}")
    logging.info(f"Hidden size: {args.hidden_size}")
    logging.info(f"Number of heads: {args.num_heads}")
    logging.info(f"Learning rate scheduler: {args.lr_scheduler_type}")
    logging.info(f"Total training steps: {num_training_steps}")
    logging.info(f"Warmup steps: {num_warmup_steps}")
    logging.info("=" * 50)
    
    if args.wandb:
        wandb.log({"trainable_parameters": trainable_params})
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.embeddings.parameters(), 'lr': args.b_lr},
        {'params': model.blocks.parameters(), 'lr': args.b_lr},
        {'params': model.classifier.parameters(), 'lr': args.h_lr}
    ], weight_decay=args.wd)
    
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_acc = 0
    total_train_time = 0
    total_eval_time = 0
    eval_num = 0
    
    for epoch in range(args.epochs):
        avg_loss, epoch_train_time = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args, epoch)
        total_train_time += epoch_train_time
        
        # Log GPU memory usage
        log_gpu_memory(args, epoch)
        
        if epoch % args.eval_epoch == 0:
            accuracy, epoch_eval_time = evaluate(model, test_loader, device, args)
            total_eval_time += epoch_eval_time
            eval_num += 1
            
            logging.info(
                f'Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, '
                f'Train time={epoch_train_time:.2f}s, Eval time={epoch_eval_time:.2f}s'
            )
            
            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "accuracy": accuracy,
                    "train_time": epoch_train_time,
                    "eval_time": epoch_eval_time,
                    "avg_epoch_train_time": total_train_time / (epoch + 1),
                    "avg_epoch_eval_time": total_eval_time / eval_num
                })
            
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), f'{args.model}_vision_best.pth')
                
    # Log final statistics
    avg_train_time = total_train_time / args.epochs
    avg_eval_time = total_eval_time / eval_num
    logging.info(
        f'Training completed. Best accuracy: {best_acc:.2f}%\n'
        f'Average training time per epoch: {avg_train_time:.2f}s\n'
        f'Average evaluation time: {avg_eval_time:.2f}s'
    )
    
    if args.wandb:
        wandb.log({
            "final/best_accuracy": best_acc,
            "final/avg_train_time": avg_train_time,
            "final/avg_eval_time": avg_eval_time
        })
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
