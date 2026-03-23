#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

model_path=/root/your_model_path/

dataset_path=$SCRIPT_DIR/data/sft_en_demo.json

output_dir=$SCRIPT_DIR/output_dir

# single gpu training
python "$SCRIPT_DIR/dft_train_ngpu.py" \
  --model_path $model_path \
  --train_json $dataset_path \
  --output_dir $output_dir \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --max_length 4096 \
  --bf16

# multi-gpu training
torchrun --nproc_per_node=2 "$SCRIPT_DIR/dft_train_ngpu.py" \
  --model_path $model_path \
  --train_json $dataset_path \
  --output_dir $output_dir \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --max_length 4096 \
  --bf16
