#!/bin/bash
# Evaluate all finished ablation runs (+ the submitted model in runs/paper/, + bicubic).
#   bash run_eval.sh <gpu id>          e.g.  bash run_eval.sh 0        (about 1-2 h on one GPU)
#   bash run_eval.sh 0 --quick         (2-5 min smoke test)
set -u
cd "$(dirname "$0")"
GPU=${1:-0}; shift || true
mkdir -p logs
echo "$(date '+%F %T')  evaluation on GPU $GPU  (log: logs/eval.log)"
CUDA_VISIBLE_DEVICES=$GPU python eval_ablation.py "$@" 2>&1 | tee logs/eval.log
echo "Results in results/ablation/"
