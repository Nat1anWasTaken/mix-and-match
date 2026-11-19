#!/usr/bin/env bash
# Run a data script offline on one Forerunner1 CPU node. Compute nodes have no internet,
# so this is the --stage process step; scripts/submit-f1.sh runs fetch before it and
# upload after it. Directly, from the repo root:
#
#   mkdir -p logs
#   sbatch scripts/f1-cpu.sbatch prepare.py --stage process --output /work1/... ...
#   sbatch scripts/f1-cpu.sbatch -m src.recipe.scorers.paloma --stage process --store ...
#
# --workers defaults to the allocated CPUs; one given after the script overrides it.
#SBATCH --job-name=data
#SBATCH --account=ACD115198
#SBATCH --partition=ct112
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# One worker process per core; keep BLAS/OpenMP/tokenizers from spawning more on top
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# No internet here: read only what --stage fetch cached, and fail fast on any download
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
# wandb writes to $WANDB_DIR/wandb/; submit-f1.sh syncs it from the login node.
# submit-f1.sh passes F1_WANDB_ROOT, and every job gets its own directory in it, so a
# rerun of the same run name never shares one with the job before it
export WANDB_MODE=offline

if [[ -n ${F1_WANDB_ROOT:-} ]]; then
    export WANDB_DIR=$F1_WANDB_ROOT/$SLURM_JOB_ID
    mkdir -p "$WANDB_DIR"
fi

# python -u <script> or python -u -m <module>; --workers goes right after it, so a
# --workers in the arguments comes later and wins
if [[ $1 == -m ]]; then
    target=(-m "$2")
    shift 2
else
    target=("$1")
    shift
fi

echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs: ${target[*]} $*"

srun uv run --offline --frozen python -u "${target[@]}" \
    --workers "$SLURM_CPUS_PER_TASK" "$@"
