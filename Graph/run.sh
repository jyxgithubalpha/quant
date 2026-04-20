#!/usr/bin/env bash
# Usage:
#   ./run.sh train 2024 3 baseline
#   ./run.sh backtest
cd "$(dirname "$0")"
mode=${1:-backtest}
case $mode in
  train)
    python train.py --year "$2" --quarter "$3" --ablation "${4:-baseline}"
    ;;
  backtest)
    python backtest.py
    ;;
  *)
    echo "unknown mode: $mode (expected: train | backtest)" >&2
    exit 1
    ;;
esac
