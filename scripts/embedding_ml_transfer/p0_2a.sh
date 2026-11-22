#!/usr/bin/env bash
# P0-2a (plan Phase 0): CompleteP LR sweep with a trainable embedding on t1-l12.
#
#   scripts/embedding_ml_transfer/p0_2a.sh main <commit> [after-job]          # 4 muon LRs
#   scripts/embedding_ml_transfer/p0_2a.sh adamw <commit> <best-muon-lr>      # adamw x1/3, x3
#   scripts/embedding_ml_transfer/p0_2a.sh transfer <commit> <best-muon-lr>   # L=12 vs 24, 2B
#
# Shared settings (plan's P0-2a notes): 1,048,576-token batches (256 x 4096), 5B =
# 4768 steps = 4,999,610,368 tokens, warmup 150 (3%), WSD decay over the last 20%,
# adamw-lr = 0.15 x muon-lr unless swept. Track 1 vocab 181,632. One node of 8 H200 per
# run, reading the recipe from the local cache.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=$1 commit=$2
RECIPE=s3://embedding-ml-transfer/recipes/multi-train-20b
CACHE=/work/u9216411/acd115198/recipe-cache
BATCH=1048576

common() {
    echo --recipe $RECIPE --recipe-cache $CACHE --mup-base-size t1-l12 \
        --vocab-size 181632 --global-batch-tokens $BATCH --micro-batch-size 8 \
        --decay-fraction 0.2 --eval fineweb_en --seed 42
}
alr() { python3 -c "print(f'{0.15 * $1:.4g}')"; }

case $mode in
main)
    after=${3:+--dependency=afterok:$3}
    for lr in 0.005 0.01 0.02 0.04; do
        name=embedding-ml-transfer-p0-2a-150m-5b-mlr$lr
        SBATCH_EXTRA=$after scripts/embedding_ml_transfer/launch.sh "$commit" "$name" 1 3 -- \
            $(common) --model-size t1-l12 --total-tokens $((4768 * BATCH)) \
            --warmup-steps 150 --muon-lr $lr --adamw-lr "$(alr $lr)"
    done ;;
adamw)
    lr=$3
    for factor in 0.3333 3; do
        a=$(python3 -c "print(f'{0.15 * $lr * $factor:.3g}')")
        name=embedding-ml-transfer-p0-2a-150m-5b-mlr$lr-alr$a
        scripts/embedding_ml_transfer/launch.sh "$commit" "$name" 1 3 -- \
            $(common) --model-size t1-l12 --total-tokens $((4768 * BATCH)) \
            --warmup-steps 150 --muon-lr $lr --adamw-lr "$a"
    done ;;
transfer)
    best=$3
    # 2B = 1907 steps; warmup 3% = 57
    for lr in $(python3 -c "print($best/2, $best, $best*2)"); do
        for size in t1-l12 t1-l24; do
            scale=$([[ $size == t1-l12 ]] && echo 150m || echo 300m)
            name=embedding-ml-transfer-p0-2a-$scale-2b-mlr$lr
            scripts/embedding_ml_transfer/launch.sh "$commit" "$name" 1 3 -- \
                $(common) --model-size $size --total-tokens $((1907 * BATCH)) \
                --warmup-steps 57 --muon-lr $lr --adamw-lr "$(alr $lr)"
        done
    done ;;
*) echo "unknown mode $mode" >&2; exit 2 ;;
esac
