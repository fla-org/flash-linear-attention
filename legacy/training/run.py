
from datasets import load_from_disk
from flame.data import DataCollatorForLanguageModeling
from flame.logging import LogCallback, get_logger
from flame.parser import get_train_args
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer

import fla  # noqa

logger = get_logger(__name__)


def main():
    args = get_train_args()
    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad token: {tokenizer.pad_token}")
    if args.from_config:
        logger.info("All model params are randomly initialized for from-scratch training.")
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_name_or_path))
    else:
        logger.info(f"Loading pretrained checkpoint {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.train()

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    logger.info(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"{tokenizer}\n{model}\n{model.config}")

    logger.info(f"Loading the `{args.split}` split directly from the cache {args.cache_dir}...")
    dataset = load_from_disk(args.cache_dir)
    logger.info(f"{dataset}")
    logger.info(f"Shuffling the dataset with seed {args.seed}")
    dataset = dataset.shuffle(seed=args.seed)
    logger.info("Creating the data collator")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, varlen=args.varlen)
    logger.info(f"{data_collator}")

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1,
        }

    trainer = Trainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        train_dataset=dataset,
    )

    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
