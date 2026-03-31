#!/bin/bash

# Default variable values
model_id=""
data_path="yahma/alpaca-cleaned"
model_dtype="float32"
device="cpu"
output_dir="./outputs"
batch_size=16
cutoff_len=1024
num_epochs=1
learning_rate=3e-4
val_set_size=0
lora_r=32
lora_alpha=16
lora_dropout=0.05
init_lora_weights="gaussian"
seed=42
quantize=""
group_texts="--group_texts"

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  -h, --help              Display this help message"
  echo "  -m, --model_id          Specify model id (required)"
  echo "  --data_path             Dataset path or HF dataset name (default: yahma/alpaca-cleaned)"
  echo "  --model_dtype           Model dtype [float32, bfloat16, float16] (default: float32)"
  echo "  --device                Device [cpu, cuda, xpu] (default: cpu)"
  echo "  --output_dir            Output directory (default: ./outputs)"
  echo "  --batch_size            Per-device train batch size (default: 16)"
  echo "  --cutoff_len            Max sequence length (default: 1024)"
  echo "  --num_epochs            Number of training epochs (default: 1)"
  echo "  --learning_rate         Learning rate (default: 3e-4)"
  echo "  --val_set_size          Validation set size, 0 to use group_texts concat (default: 0)"
  echo "  --lora_r                LoRA rank (default: 32)"
  echo "  --lora_alpha            LoRA alpha (default: 16)"
  echo "  --lora_dropout          LoRA dropout (default: 0.05)"
  echo "  --init_lora_weights     LoRA init method [gaussian, olora, etc.] (default: gaussian)"
  echo "  --seed                  Random seed (default: 42)"
  echo "  --quantize              Enable 4-bit quantization"
  echo "  --group_texts           Enable group_texts concatenation (default)"
  echo "  --no_group_texts        Disable group_texts concatenation"
  echo "  --max_steps             Max training steps, -1 for full epoch (default: -1)"
}

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

handle_options() {
  while [ $# -gt 0 ]; do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      -m | --model_id*)
        if ! has_argument "$@"; then
          echo "Model ID not specified." >&2
          usage
          exit 1
        fi
        model_id=$(extract_argument "$@")
        shift
        ;;
      --data_path*)
        data_path=$(extract_argument "$@")
        shift
        ;;
      --model_dtype*)
        model_dtype=$(extract_argument "$@")
        shift
        ;;
      --device*)
        device=$(extract_argument "$@")
        shift
        ;;
      --output_dir*)
        output_dir=$(extract_argument "$@")
        shift
        ;;
      --batch_size*)
        batch_size=$(extract_argument "$@")
        shift
        ;;
      --cutoff_len*)
        cutoff_len=$(extract_argument "$@")
        shift
        ;;
      --num_epochs*)
        num_epochs=$(extract_argument "$@")
        shift
        ;;
      --learning_rate*)
        learning_rate=$(extract_argument "$@")
        shift
        ;;
      --val_set_size*)
        val_set_size=$(extract_argument "$@")
        shift
        ;;
      --lora_r*)
        lora_r=$(extract_argument "$@")
        shift
        ;;
      --lora_alpha*)
        lora_alpha=$(extract_argument "$@")
        shift
        ;;
      --lora_dropout*)
        lora_dropout=$(extract_argument "$@")
        shift
        ;;
      --init_lora_weights*)
        init_lora_weights=$(extract_argument "$@")
        shift
        ;;
      --seed*)
        seed=$(extract_argument "$@")
        shift
        ;;
      --quantize)
        quantize="--quantize"
        ;;
      --no_group_texts)
        group_texts="--no_group_texts"
        ;;
      --group_texts)
        group_texts="--group_texts"
        ;;
      --max_steps*)
        max_steps=$(extract_argument "$@")
        shift
        ;;
      *)
        echo "Invalid option: $1" >&2
        usage
        exit 1
        ;;
    esac
    shift
  done
}

handle_options "$@"

if [ -z "$model_id" ]; then
  echo "Error: --model_id is required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
file="${SCRIPT_DIR}/olora_finetuning.py"

# Auto-generate accelerate config (always, based on current machine)
num_nodes=$(lscpu | grep -oP 'NUMA node\(s\):\s+\K\d+' || echo 1)

auto_hostfile="${SCRIPT_DIR}/.auto_hostfile"
: > "$auto_hostfile"
for ((i=0; i<num_nodes; i++)); do
  echo "127.0.0.1" >> "$auto_hostfile"
done

auto_config="${SCRIPT_DIR}/.auto_cpu_config.yaml"
cat > "$auto_config" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_CPU
enable_cpu_affinity: false
machine_rank: 0
main_process_ip: 127.0.0.1
main_process_port: 29500
main_training_function: main
mixed_precision: 'bf16'
mpirun_config:
  mpirun_ccl: '0'
  mpirun_hostfile: "${auto_hostfile}"
num_machines: 1
num_processes: ${num_nodes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
EOF
accelerate_config="$auto_config"
echo "[INFO] accelerate config: num_processes=${num_nodes}, hostfile=${auto_hostfile}"

accelerate launch --config_file "$accelerate_config" "$file" \
  --base_model "$model_id" \
  --data_path "$data_path" \
  --output_dir "$output_dir" \
  --batch_size "$batch_size" \
  --cutoff_len "$cutoff_len" \
  --num_epochs "$num_epochs" \
  --learning_rate "$learning_rate" \
  --val_set_size "$val_set_size" \
  --lora_r "$lora_r" \
  --lora_alpha "$lora_alpha" \
  --lora_dropout "$lora_dropout" \
  --init_lora_weights "$init_lora_weights" \
  --seed "$seed" \
  --dtype "$model_dtype" \
  --device_map "$device" \
  $quantize \
  $group_texts \
  --max_steps "$max_steps"