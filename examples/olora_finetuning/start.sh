#!/bin/bash

MODELS=(
  /home/models/qwen-3-8b
  /home/models/qwen-3-14b
  /home/models/qwen-3-32b
  /home/models/llama-2-13b
  /home/models/llama-2-7b
)
CUTOFF_LENS=(1024 512)
BATCH_SIZES=(16 4)
# GROUP_TEXTS_OPTS=("--group_texts" "--no_group_texts")
GROUP_TEXTS_OPTS=("--group_texts")

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for model in "${MODELS[@]}"; do
  for cutoff in "${CUTOFF_LENS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for gt in "${GROUP_TEXTS_OPTS[@]}"; do
        model_name=$(basename "$model")
        if [[ "$gt" == "--group_texts" ]]; then
          gt_label="concat"
        else
          gt_label="noconcat"
        fi
        log_file="${LOG_DIR}/${model_name}_cutoff${cutoff}_bs${bs}_${gt_label}.log"

        if [[ -f "$log_file" && -s "$log_file" ]]; then
          echo "[SKIP] $log_file already exists, skipping..."
          continue
        fi

        echo "======================================"
        echo "Model: $model_name | cutoff_len: $cutoff | batch_size: $bs | $gt_label"
        echo "Log: $log_file"
        echo "======================================"

        bash run_lora.sh -m "$model" \
          --data_path /home/finetune_code/alpaca_data.json \
          --model_dtype bfloat16 \
          --accelerate_config /root/fine-tune/cpu_config.yaml \
          --cutoff_len "$cutoff" \
          --batch_size "$bs" \
          $gt \
          > "$log_file" 2>&1
      done
    done
  done
done