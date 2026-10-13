#!/bin/bash

source "$(dirname "$0")/common.sh"
for E in $OUT/spanwm/*.evalset.jsonl; do
  B=${E%.evalset.jsonl}; NAME=$(basename $B)
  [ -e $B.judge.summary.json ] && continue
  $PY evaluation/judge_similarity.py --input $E --output $B.judge.jsonl --reference-field draft \
      --limit 100 --workers 8 --resume > $OUT/logs/judge_$NAME.log 2>&1 || echo "FAILED judge $NAME"
done
