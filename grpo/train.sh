#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

model_path=/root/your_model_path/

train_dataset_path=$SCRIPT_DIR/data/gsm8k_train_1of8.json

eval_dataset_path=$SCRIPT_DIR/data/gsm8k_test_1of8.json

output_dir=$SCRIPT_DIR/output_dir

# single gpu training
python "$SCRIPT_DIR/grpo_train.py" \
  --model_path $model_path \
  --train_json $train_dataset_path \
  --eval_json $eval_dataset_path \
  --eval_steps 50 \
  --output_dir $output_dir \
  --train_batch_size 16 \
  --ppo_mini_batch_size 4 \
  --ppo_micro_batch_size_per_gpu 1 \
  --num_train_epochs 2 \
  --kl_coef 0.01 \
  --bf16
