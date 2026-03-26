#!/bin/bash
# Parse finetune logs and output a summary table
# Usage: place in logs/ directory, run: bash parse_logs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf "%-30s %10s %5s %10s %8s %12s\n" "Model" "SeqLen" "BS" "Concat" "Steps" "Time(min)"
printf "%s\n" "--------------------------------------------------------------------------------"

for log in "$SCRIPT_DIR"/*.log; do
  [ -f "$log" ] || continue
  fname=$(basename "$log" .log)

  # Parse filename: {model}_cutoff{len}_bs{size}_{concat|noconcat}
  model=$(echo "$fname" | sed 's/_cutoff[0-9].*$//')
  cutoff=$(echo "$fname" | grep -oP 'cutoff\K[0-9]+')
  bs=$(echo "$fname" | grep -oP 'bs\K[0-9]+')
  concat=$(echo "$fname" | grep -oP '(concat|noconcat)$')

  # Extract steps and time from progress bar: 100%|...| 407/407 [H:MM:SS<...]
  steps=$(grep -oP '100%\|[^|]*\| \K\d+' "$log" | tail -1)
  time_str=$(grep -oP '100%\|[^|]*\| \d+/\d+ \[\K[0-9:]+' "$log" | tail -1)

  if [ -n "$time_str" ]; then
    # Split H:M:S or M:S
    IFS=: read -ra parts <<< "$time_str"
    if [ ${#parts[@]} -eq 3 ]; then
      minutes=$(echo "${parts[0]} * 60 + ${parts[1]} + ${parts[2]} / 60" | bc -l)
    elif [ ${#parts[@]} -eq 2 ]; then
      minutes=$(echo "${parts[0]} + ${parts[1]} / 60" | bc -l)
    else
      minutes="N/A"
    fi
    if [ "$minutes" != "N/A" ]; then
      minutes=$(printf "%.1f" "$minutes")
    fi
  else
    minutes="N/A"
  fi

  printf "%-30s %10s %5s %10s %8s %12s\n" "$model" "$cutoff" "$bs" "$concat" "${steps:-N/A}" "$minutes"
done | sort
