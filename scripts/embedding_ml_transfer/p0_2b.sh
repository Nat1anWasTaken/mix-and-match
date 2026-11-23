#!/usr/bin/env bash
# P0-2b (plan Phase 0): CompleteP scale calibration with a frozen donor embedding on t1-l12.
#
#   scripts/embedding_ml_transfer/p0_2b.sh qwen <commit> <node> <emb-mult> <out-mult> [muon-lr]
#   scripts/embedding_ml_transfer/p0_2b.sh oracle <commit> <node> <oracle_emb.pt> <emb-mult> <out-mult> [muon-lr]
#
# Same data, batch and schedule as the P0-2a main sweep (Multi-Train v2, 1,048,576-token
# batches, 5B = 4768 steps, warmup 150, WSD decay 0.2), so each point compares directly
# with P0-2a's trainable-embedding run at the same muon-lr. The embedding is frozen, so
# adamw-lr only moves the RMSNorm gains; it stays at P0-2a's 0.15 x muon-lr.
#
# Why the multipliers: logits are h . E (tied), so a donor with a small row RMS gives
# small logits. P0-2a's trained embedding ends near RMS 0.18 (mlr0.04), Qwen3-0.6B-Base's
# is 0.0302: scaling both sides by s is the same as loading s x E, and s = 6 matches
# the trained scale. emb-mult and out-mult are swept separately to tell the two apart.
set -euo pipefail
cd "$(dirname "$0")/../.."
track=$1 commit=$2 node=$3
shift 3
if [[ $track == qwen ]]; then
    donor=/work/u9216411/acd115198/embedding-ml-transfer-runs/donors/qwen3-0.6b-base-emb.pt
    vocab=151936
else
    donor=$1
    vocab=181632
    shift
fi
emb=$1 out=$2 lr=${3:-0.04}
alr=$(python3 -c "print(f'{0.15 * $lr:.4g}')")
name=embedding-ml-transfer-p0-2b-150m-5b-$track-mlr$lr-ein$emb-eout$out
BATCH=1048576
SBATCH_EXTRA="--nodelist=25a-hgpn$node" scripts/embedding_ml_transfer/launch.sh "$commit" "$name" 1 3 -- \
    --recipe s3://embedding-ml-transfer/recipes/multi-train-20b-v2 \
    --recipe-cache /work/u9216411/acd115198/recipe-cache --mup-base-size t1-l12 \
    --vocab-size $vocab --global-batch-tokens $BATCH --micro-batch-size 8 \
    --decay-fraction 0.2 --eval fineweb_en --seed 42 --model-size t1-l12 \
    --total-tokens $((4768 * BATCH)) --warmup-steps 150 --muon-lr $lr --adamw-lr "$alr" \
    --donor-embedding "$donor" --freeze-embeddings \
    --embedding-multiplier "$emb" --output-multiplier "$out"
echo "$name"
