#!/usr/bin/env bash
# Phase 1 (route A) training runs, 1 node x 8 H200 each:
#
#   scripts/embedding_ml_transfer/phase1.sh <experiment> <commit> <node>
#
# <experiment> picks the settings below; <node> pins the job (empty: let Slurm pick).
# Shared by every experiment unless it says otherwise:
#   - t1-l24 (300M), --mup-base-size t1-l12, Multi-Train v2, 19,073 steps = 19.999B
#     tokens (the same 99.0% of the recipe as B0-Lower), warmup 572, WSD decay 0.2,
#     batch 1,048,576, seed 42, track-1 vocab 181,632
#   - HP_frozen_oracle (P0-2b): muon-lr 0.04, adamw-lr 0.006, embedding and output
#     multipliers 0.5, Oracle_Emb (B0-Upper-r2 step-22888 resume.pt) frozen and tied
#   - the training-time masked softmax over the ids counted > 0 in Multi-Train v2
#     (plan Phase 1 revision): the L_ban rows are never a negative
set -euo pipefail
cd "$(dirname "$0")/../.."
experiment=$1 commit=$2 node=${3:-}
RUNS=/work/u9216411/acd115198/embedding-ml-transfer-runs
BATCH=1048576
model=t1-l24 scale=300m steps=19073 warmup=572 tokens=20b hours=12
recipe=multi-train-20b-v2
keep=$RUNS/softmax-keep/multi-train-20b-v2.npz
donor=$RUNS/donors/oracle_emb.pt vocab=181632
lr=0.04 alr=0.006 ein=0.5 eout=0.5 seed=42
embedding=(--donor-embedding "$donor" --freeze-embeddings)
variant=""
extra=()

trainable=(--donor-embedding "$donor" --adamw-weight-decay 0)
case $experiment in
a1)
    ;;
a1-seed2)
    # Plan §6: the key comparisons (A1 vs Floor, A1 vs A4c) get a second seed. The seed
    # changes the body's init; the data order is the recipe's and does not move
    experiment=a1 variant=-seed2 seed=43
    ;;
a2)
    # Mono-Train (English only, 20.20B); its own keep set, which also drops es/de/ru-only ids
    recipe=mono-train-20b-v2 keep=$RUNS/softmax-keep/mono-train-20b-v2.npz
    ;;
a3)
    # Track 2: Qwen3-0.6B-Base's embedding at HP_frozen_qwen, vocab 151,936 (no
    # Fake-French rows); Multi-Train has no id >= 151,936, so A1's keep set applies as is
    donor=$RUNS/donors/qwen3-0.6b-base-emb.pt vocab=151936 ein=1 eout=1
    embedding=(--donor-embedding "$donor" --freeze-embeddings)
    ;;
a4c)
    # Oracle_Emb as init, trainable, embedding wd 0 (--adamw-weight-decay only reaches the
    # tied embedding; RMSNorm gains are never decayed), masked softmax: the rows outside the
    # keep set (all L_ban rows among them) get no gradient and no decay, so they stay
    # Oracle_Emb bit for bit and only the L_train rows drift. A1's multipliers, so A4c vs A1
    # differs only in the L_train rows being trainable
    embedding=("${trainable[@]}")
    ;;
a4c-seed2)
    # Plan §6 (revised 2026-11-24): A1 vs A4c is a key comparison, paired with A1-seed2
    embedding=("${trainable[@]}") experiment=a4c variant=-seed2 seed=43
    ;;
a4a)
    # As A4c with the full softmax: the L_ban rows are trainable too and, as negatives,
    # pushed every step. A4a vs A4c is one switch, the mask
    embedding=("${trainable[@]}") keep=""
    ;;
a4b)
    # As A4a with the default embedding wd 0.1
    embedding=(--donor-embedding "$donor") keep=""
    ;;
a1-mask150)
    # The masked arm of the 150M ablation; the full-softmax arm is P0-2b's
    # embedding-ml-transfer-p0-2b-150m-5b-oracle-mlr0.04-ein0.5-eout0.5 (same settings)
    model=t1-l12 scale=150m steps=4768 warmup=150 tokens=5b hours=3 variant=-mask
    ;;
*)
    echo "unknown experiment $experiment" >&2
    exit 1
    ;;
esac

name=embedding-ml-transfer-$experiment-$scale-$tokens$variant${SUFFIX:-}
SBATCH_EXTRA="${node:+--nodelist=25a-hgpn$node}" scripts/embedding_ml_transfer/launch.sh "$commit" "$name" 1 $hours -- \
    --recipe s3://embedding-ml-transfer/recipes/$recipe \
    --recipe-cache /work/u9216411/acd115198/recipe-cache --mup-base-size t1-l12 \
    --vocab-size $vocab --global-batch-tokens $BATCH --micro-batch-size 8 \
    --decay-fraction 0.2 --eval fineweb_en --seed $seed --model-size $model \
    --total-tokens $((steps * BATCH)) --warmup-steps $warmup --muon-lr $lr --adamw-lr $alr \
    "${embedding[@]}" --embedding-multiplier $ein --output-multiplier $eout \
    ${keep:+--softmax-keep "$keep"} "${extra[@]}"
echo "$name"
