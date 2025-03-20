args=$@
for arg in $args; do
    eval "$arg"
done

echo "model:            ${model:=fla-hub/gla-1.3B-100B}"
echo "tokenizer:        ${tokenizer:=fla-hub/gla-1.3B-100B}"
echo "project:          ${project:=fla}"
echo "type:             ${type:=gla}"
echo "data:             ${data:=}"
echo "name:             ${name:=}"
echo "cache:            ${cache:=}"
echo "varlen:           ${varlen:=false}"
echo "seed:             ${seed:=42}"
echo "context:          ${context:=2048}"
echo "steps:            ${steps:=0}"
echo "save:             ${save:=2048}"
echo "limit:            ${limit:=1}"
echo "preprocessing:    ${preprocessing:=32}"
echo "workers:          ${workers:=32}"
echo "prefetch:         ${prefetch:=2}"
echo "logging:          ${logging:=32}"
echo "config:           ${config:=configs/deepspeed.yaml}"

echo "lr:               ${lr:=3e-4}"
echo "scheduler:        ${scheduler:=cosine_with_min_lr}"
echo "epochs:           ${epochs:=1}"
echo "optim:            ${optim:=adamw_torch_fused}"
echo "decay:            ${decay:=0.01}"
echo "beta1:            ${beta1:=0.9}"
echo "beta2:            ${beta2:=0.95}"
echo "norm:             ${norm:=1.0}"
echo "batch:            ${batch:=32}"
echo "update:           ${update:=1}"
echo "warmup:           ${warmup:=512}"
echo "path:             ${path:=}"
echo "checkpoint:       ${checkpoint:=}"
echo "node:             ${node:=}"
echo "rank:             ${rank:=}"
echo "ip:               ${ip:=}"
echo "port:             ${port:=}"
echo "nodes:            ${nodes:=1}"
echo "gpus:             ${gpus:=8}"

params="--model_name_or_path $model \
    --tokenizer $tokenizer \
    --use_fast_tokenizer \
    --do_train \
    --dataset $data \
    --context_length $context \
    --preprocessing_num_workers $preprocessing \
    --dataloader_num_workers $workers \
    --dataloader_prefetch_factor $prefetch \
    --output_dir $path \
    --overwrite_output_dir \
    --logging_steps $logging \
    --include_num_input_tokens_seen \
    --save_steps $save \
    --save_total_limit $limit \
    --learning_rate $lr \
    --lr_scheduler_type $scheduler \
    --warmup_steps $warmup \
    --optim $optim \
    --weight_decay $decay \
    --adam_beta1=$beta1 \
    --adam_beta2=$beta2 \
    --max_grad_norm $norm \
    --num_train_epochs $epochs \
    --per_device_train_batch_size $batch \
    --gradient_accumulation_steps $update \
    --seed $seed \
    --logging_steps $logging \
    --log_level info \
    --bf16"

if [ $steps -gt 0 ]; then
    params+=" --max_steps $steps"
fi

if [ "$name" != "" ]; then
  params+=" --dataset_name $name"
fi
if [ "$cache" != "" ]; then
  params+=" --cache_dir $cache"
fi
if [ "$varlen" == "true" ]; then
  params+=" --varlen"
fi
if [ "$checkpoint" != "" ]; then
  params+=" --resume_from_checkpoint $checkpoint"
fi
if [ "$WANDB_DISABLED" != "true" ]; then
  params+=" --report_to wandb \
  --run_name $type.$(basename $path)"
else
  params+=" --report_to none"
fi

echo "Launching training..."
accelerate_params=""
if [ "$rank" != "" ]; then
  accelerate_params+=" --machine_rank $rank  \
    --num_processes $((nodes * gpus)) \
    --num_machines $nodes \
    --main_process_ip $ip \
    --main_process_port $port \
    --same_network"
fi

if [[ $config == *"deepspeed"* ]]; then
cat <<EOF > "configs/ds_config.json"
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": false,
    "contiguous_gradients": true
  }
}
EOF
cat <<EOF > $config
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: configs/ds_config.json
  zero3_init_flag: true
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: $gpus
use_cpu: false
EOF
fi
if [[ $config == *"fsdp"* ]]; then
cat <<EOF > $config
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: HYBRID_SHARD_ZERO2
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: $nodes
num_processes: $((nodes * gpus))
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

cat $config

set -x
mkdir -p $path
cp * $path
cp -r configs $path
cp -r flame   $path
cp -r ../fla $path

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [ "$date" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi
export WANDB_RESUME=allow
export WANDB_NAME="$type.$(basename $path)"
export WANDB_PROJECT=$project
export WANDB_RUN_ID="$WANDB_NAME-$date"
accelerate launch $accelerate_params --config_file $config run.py $params

echo "RUNNING DONE!"
