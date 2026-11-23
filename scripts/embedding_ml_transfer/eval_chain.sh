#!/usr/bin/env bash
# Queue the Phase 1 evaluations of one run behind its training job:
#
#   scripts/embedding_ml_transfer/eval_chain.sh <train-job> <run-name> <step> <out-dir> [controls]
#
# Always: the raw checkpoint (full softmax, with the within-language / language-selection
# split). With "controls": also the self-Floor (--floor --floor-keep: this run's body,
# every row outside the training keep set at the mean) and the shuffled control
# (--shuffle-seed 0..4: L_ban rows permuted within each language). Outputs go to
# <out-dir>/<run-name>{,-self-floor,-shuffle-s<seed>}.{json,npz}.
set -euo pipefail
cd "$(dirname "$0")/../.."
job=$1 name=$2 step=$3 out=$4 controls=${5:-}
RUNS=/work/u9216411/acd115198/embedding-ml-transfer-runs
KEEP=$RUNS/softmax-keep/multi-train-20b-v2.npz
mkdir -p "$out"
common=(--checkpoint "$RUNS/checkpoints/$name/step-$step" --fake-french "$RUNS/p0-1/fake-french"
    --language-sets "$RUNS/language-sets" --flores "$RUNS/eval-data/flores"
    --fineweb2 "$RUNS/eval-data/fineweb2")
submit() {
    local tag=$1
    shift
    sbatch --parsable --dependency=afterok:$job --job-name="eval-$name$tag" \
        scripts/embedding_ml_transfer/evaluate.sbatch scripts/embedding_ml_transfer/evaluate.py \
        "${common[@]}" --output "$out/$name$tag" "$@"
}
echo "raw $(submit "")"
if [[ $controls == controls ]]; then
    echo "self-floor $(submit -self-floor --floor --floor-keep "$KEEP")"
    for seed in 0 1 2 3 4; do
        echo "shuffle-s$seed $(submit -shuffle-s$seed --shuffle-seed $seed)"
    done
fi
