#!/usr/bin/env bash
# Block until every named kernel reaches a terminal state; prints one line per kernel as it finishes.
# Usage: bash wait_done.sh slug1 slug2 ...    (gives up after ~6 h)
K=~/yodran/kaggle-quantum-compile/.venv/bin/kaggle
pending=("$@")
for _ in $(seq 1 240); do
  left=()
  for k in "${pending[@]}"; do
    s=$($K kernels status "sundarshahi/$k" 2>&1)
    if [[ $s == *COMPLETE* || $s == *ERROR* || $s == *CANCEL* ]]; then
      echo "$(date +%H:%M) $k ${s##*status }"
    else
      left+=("$k")
    fi
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -eq 0 ] && exit 0
  sleep 90
done
echo "$(date +%H:%M) TIMED OUT: ${pending[*]}"
