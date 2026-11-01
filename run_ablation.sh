#!/bin/bash
# =============================================================================================
# Train the six ablation runs (A1 A2 A3 B1 B2 C1). Run from this folder, inside tmux.
#
#   bash run_ablation.sh <GPUS_PER_RUN> <free GPU ids, comma-separated>
#
#   bash run_ablation.sh 1 0,1,2,3,4,5      -> 6 runs at once, one GPU each
#   bash run_ablation.sh 1 0,1,2            -> 3 at a time, 2 rounds
#   bash run_ablation.sh 4 0,1,2,3,4,5,6,7  -> 2 at a time with 4 GPUs each, 3 rounds
#
# Safe to run again after any interruption: finished runs are skipped, unfinished runs
# RESUME from their last epoch (runs/<name>_<date>/last.pth).
# Logs: logs/<run>.log      Results: runs/<run>_<date-time>/
# =============================================================================================
set -u
cd "$(dirname "$0")"
GPR=${1:?usage: bash run_ablation.sh <GPUS_PER_RUN> <gpu ids, e.g. 0,1,2,3>}
IFS=',' read -ra G <<< "${2:?give the free GPU ids, e.g. 0,1,2,3}"
RUNS=(${RUNS_OVERRIDE:-A1_full_w111 A2_full_w039_18 A3_full_w003 B1_single_guidance_w003 B2_oneshot_w003 C1_stn_off_w111})
SLOTS=$(( ${#G[@]} / GPR ))
[ "$SLOTS" -ge 1 ] || { echo "Need at least $GPR GPU ids for $GPR GPUs per run."; exit 1; }
mkdir -p logs
echo "$(date '+%F %T')  ${#RUNS[@]} runs, $GPR GPU(s) each, $SLOTS at a time, GPUs ${2}"

todo=()
for c in "${RUNS[@]}"; do
  if ls -d runs/${c}_*/DONE >/dev/null 2>&1; then echo "  skip $c (already finished)"; else todo+=("$c"); fi
done

i=0
while [ $i -lt ${#todo[@]} ]; do
  for ((s=0; s<SLOTS && i<${#todo[@]}; s++)); do
    c=${todo[$i]}; i=$((i+1))
    devs=$(IFS=,; echo "${G[*]:$((s*GPR)):$GPR}")
    cfg=configs/ablation/$c.yaml
    last=$(ls -t runs/${c}_*/last.pth 2>/dev/null | head -1)
    if [ -n "$last" ]; then                       # unfinished earlier attempt -> resume it
      cfg=logs/${c}_resume.yaml
      cp configs/ablation/$c.yaml "$cfg"
      echo "resume_path: \"$(pwd)/$last\"" >> "$cfg"
      echo "$(date '+%F %T')  RESUME $c from $last on GPU(s) $devs"
    else
      echo "$(date '+%F %T')  START  $c on GPU(s) $devs"
    fi
    CUDA_VISIBLE_DEVICES=$devs python train.py --config "$cfg" >> logs/$c.log 2>&1 &
    sleep 5
  done
  wait
done
echo "$(date '+%F %T')  all runs finished. Next:  bash run_eval.sh <one free GPU id>"
for c in "${RUNS[@]}"; do
  if ls -d runs/${c}_*/DONE >/dev/null 2>&1; then echo "  OK      $c"; else echo "  FAILED  $c   (see logs/$c.log; run this script again to resume)"; fi
done
