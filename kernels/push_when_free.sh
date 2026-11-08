#!/usr/bin/env bash
# Push the given kernel folders, retrying while Kaggle's 5-CPU-session limit is reached.
# Usage: bash push_when_free.sh slug1 slug2 ...   (prints one line per push outcome)
cd "$(dirname "$0")/build" || exit 1
K=~/yodran/kaggle-quantum-compile/.venv/bin/kaggle
pending=("$@")
for _ in $(seq 1 120); do
  left=()
  for k in "${pending[@]}"; do
    out=$($K kernels push -p "$k" 2>&1 | tail -1)
    if [[ $out == *successfully* ]]; then echo "$(date +%H:%M) PUSHED $k"; else left+=("$k"); fi
    [[ $out == *successfully* || $out == *"session count"* ]] || echo "$(date +%H:%M) ERROR $k: $out"
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -eq 0 ] && exit 0
  sleep 120
done
echo "$(date +%H:%M) GAVE UP: ${pending[*]}"
