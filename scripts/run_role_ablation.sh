#!/bin/bash

source "$(dirname "$0")/common.sh"
[ $# -gt 0 ] && MODELS=("$@")
mkdir -p $OUT/role_ablation
for MODEL in "${MODELS[@]}"; do
  TAG=${MODEL##*/}; D=$OUT/role_ablation/${TAG}_c4; L=$OUT/logs/role_ablation_${TAG}_c4.log
  $PY experiments/role_ablation.py run --input dataset/prompts/c4_n200.jsonl --output-dir $D \
      --model $MODEL --seed 42 > $L 2>&1 \
    && $PY experiments/role_ablation.py summarize --run-dir $D --bootstrap-samples 10000 --seed 123 >> $L 2>&1 \
    || echo "FAILED role_ablation $TAG (see $L)"
done
