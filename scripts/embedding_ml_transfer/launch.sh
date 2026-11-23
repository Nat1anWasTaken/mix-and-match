#!/usr/bin/env bash
# Launch one embedding-ml-transfer training run on nano4 from a frozen copy of the code:
#
#   scripts/embedding_ml_transfer/launch.sh <commit> <run-name> <nodes> <hours> -- <train.py args>
#
# The run trains from a detached worktree of <commit> under $RUNS/code/<commit>, so
# editing or switching branches here never changes a queued or running job. Checkpoints
# go to $RUNS/checkpoints/<run-name>, the log to $RUNS/logs/<run-name>-<job>.out.
# --run-name, --output-dir, --wandb-project and --tokenizer are added here.
set -euo pipefail

RUNS=/work/u9216411/acd115198/embedding-ml-transfer-runs
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$REPO" rev-parse --short=7 "$1")
name=$2 nodes=$3 hours=$4
shift 4
[[ "${1:-}" == "--" ]] && shift

code=$RUNS/code/$commit
if [[ ! -d $code ]]; then
    git -C "$REPO" worktree add -q --detach "$code" "$commit"
    (cd "$code" && uv sync --frozen -q)
fi

cd "$code"
sbatch ${SBATCH_EXTRA:-} --job-name="$name" --nodes="$nodes" --gpus-per-node=H200:8 --cpus-per-task=96 \
    --mem=1000G --time="$hours:00:00" --exclude='25a-hgpn[069,141,173-176]' \
    --output="$RUNS/logs/$name-%j.out" scripts/train.sbatch \
    --run-name "$name" --output-dir "$RUNS/checkpoints" \
    --wandb-project embedding-ml-transfer --tokenizer Qwen/Qwen3-0.6B-Base "$@"
