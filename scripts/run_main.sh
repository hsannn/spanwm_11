#!/bin/bash

source "$(dirname "$0")/common.sh"
[ $# -gt 0 ] && MODELS=("$@")
[ -e dataset/prompts/c4_n200.jsonl ] || $PY prepare_prompts.py
for MODEL in "${MODELS[@]}"; do
  TAG=${MODEL##*/}
  for DS in "${DATASETS[@]}"; do
    F=$OUT/spanwm/${TAG}_${DS}.jsonl
    [ -e ${F%.jsonl}.scores.json ] && continue
    L=$OUT/logs/spanwm_${TAG}_${DS}.log
    $PY generate.py --input dataset/prompts/${DS}_n200.jsonl --output $F --model $MODEL > $L 2>&1 \
      && $PY detect.py --input $F --output ${F%.jsonl}.scores.json >> $L 2>&1 \
      || echo "FAILED $TAG $DS (see $L)"
  done
done
