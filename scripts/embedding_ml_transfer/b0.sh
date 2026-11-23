#!/usr/bin/env bash
# B0-Upper / B0-Lower (plan Phase 0): t1-l24 (300M) with a trainable, randomly initialized
# tied embedding, at P0-2a's HP_trainable, on 1 node x 8 H200.
#
#   scripts/embedding_ml_transfer/b0.sh upper <commit> <node> <muon-lr> <adamw-lr>
#   scripts/embedding_ml_transfer/b0.sh lower <commit> <node> <muon-lr> <adamw-lr>
#
# Lower reads Multi-Train v2 (20.20B) for 19,073 steps = 19.999B tokens. Upper reads
# Oracle-Train v2 (24.24B = Multi-Train + zh 2.02B + fake_fra 2.02B) for 22,888 steps =
# 24.000B: the same 99.0% of each recipe, and RecipeTokenStream draws every cursor in
# proportion, so both see the same L_train tokens and Upper only adds L_ban. Warmup 3%,
# WSD decay over the last 20%, batch 1,048,576 tokens, track-1 vocab 181,632 for both.
# Lower trains with the masked softmax (--softmax-keep), Upper with the full one.
set -euo pipefail
cd "$(dirname "$0")/../.."
which=$1 commit=$2 node=$3 lr=$4 alr=$5
BATCH=1048576
if [[ $which == upper ]]; then
    recipe=oracle-train-24b-v2 steps=22888 warmup=687 tokens=24b
else
    recipe=multi-train-20b-v2 steps=19073 warmup=572 tokens=20b
    # Masked softmax (plan Phase 1 revision, B0-Lower only): logits over the ids counted
    # > 0 in Multi-Train v2 (token_counts.py), L_ban rows stay at their init values
    extra="--softmax-keep /work/u9216411/acd115198/embedding-ml-transfer-runs/softmax-keep/multi-train-20b-v2.npz"
fi
name=embedding-ml-transfer-b0-$which-300m-$tokens${B0_SUFFIX:-}
SBATCH_EXTRA="${node:+--nodelist=25a-hgpn$node}" scripts/embedding_ml_transfer/launch.sh "$commit" "$name" 1 12 -- \
    --recipe s3://embedding-ml-transfer/recipes/$recipe \
    --recipe-cache /work/u9216411/acd115198/recipe-cache --mup-base-size t1-l12 \
    --vocab-size 181632 --global-batch-tokens $BATCH --micro-batch-size 8 \
    --decay-fraction 0.2 --eval fineweb_en --seed 42 --model-size t1-l24 \
    --total-tokens $((steps * BATCH)) --warmup-steps $warmup --muon-lr "$lr" --adamw-lr "$alr" ${extra:-}
echo "$name"
