#!/bin/bash
# Harmony donor-batch-correction vs the perslice baseline, leave-one-donor-out on
# all 3 donors. Tests whether Harmony closes the cross-donor generalization gap.
# 6 runs = {perslice, harmony} x {test S1, S2, S3}, 100 epochs, seed 0.
set -u
PY="${PY:-python}"                         # override with: PY=/path/to/venv/python
REPO="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUTF8=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
LOG=$REPO/results/harmony_lodo_run.log
DONE=$REPO/results/harmony_lodo_DONE.txt
rm -f "$DONE"; : > "$LOG"
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run(){ # mode trainpairs testpair foldtag
  local mode=$1 tr=$2 te=$3 tag=$4 out="sutura_lodo_${1}_test${4}"
  log "START $out  (test $te)"
  if $PY "$REPO/train.py" --train-pairs "$tr" --test-pair "$te" \
        --feature-mode "$mode" --epochs 100 --steps-per-epoch 24 --seed 0 \
        --out "$out" --data-dir "$REPO/data" --results-dir "$REPO/results" >> "$LOG" 2>&1; then
    log "OK   $out"; else log "FAIL $out (exit $?)"; fi
}

S1=151507/151508; S2=151669/151670; S3=151673/151674
log "=== HARMONY vs PERSLICE LODO START (6 runs) ==="
for mode in perslice harmony; do
  run "$mode" "$S2,$S3" "$S1" S1 &
  run "$mode" "$S1,$S3" "$S2" S2 &
  run "$mode" "$S1,$S2" "$S3" S3 &
  wait
done
F=$(grep -c FAIL "$LOG")
log "=== COMPLETE ($F failures) ==="
echo "harmony-lodo finished $(date -u), $F failures" > "$DONE"
