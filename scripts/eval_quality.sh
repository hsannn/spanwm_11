#!/bin/bash

source "$(dirname "$0")/common.sh"
for S in $OUT/spanwm/*.scores.json; do
  B=${S%.scores.json}; F=$B.jsonl; NAME=$(basename $B)
  [ -e $B.ppl.json ] || $PY evaluation/eval_ppl.py --input $F --output $B.ppl.json \
      > $OUT/logs/ppl_$NAME.log 2>&1 || echo "FAILED ppl $NAME"
  [ -e $B.evalset.jsonl ] || $PY evaluation/filter_empty.py --input $F --output $B.evalset.jsonl \
      --fields draft watermarked_text > $OUT/logs/filter_$NAME.log 2>&1 || echo "FAILED filter $NAME"
  [ -e $B.cos.json ] || $PY evaluation/eval_similarity.py --input $B.evalset.jsonl --output $B.cos.json \
      --reference-field draft > $OUT/logs/cos_$NAME.log 2>&1 || echo "FAILED cos $NAME"
done
