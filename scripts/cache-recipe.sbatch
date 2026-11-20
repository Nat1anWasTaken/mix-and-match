#!/usr/bin/env bash
# Copy a compiled recipe and the shards it reads from S3 to /work, for train.py
# --recipe-cache (see src/recipe/cache.py). All arguments go to src.recipe.cache:
#
#   mkdir -p logs
#   sbatch scripts/cache-recipe.sbatch --recipe s3://mix-and-match/recipes/dolma-v1_7-30b \
#       --cache /work/u9216411/acd115198/recipe-cache
#
# Flock S3 gives about 0.85 GB/s, so a 30B-token recipe (~140 GB) takes a few minutes.
# Files already cached are skipped, so if the time limit runs out, submit it again.
# The copy needs no GPU, but the H200 partitions require one and allow 12 CPUs per GPU
#SBATCH --job-name=cache-recipe
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Same S3 setup as train.sbatch: the flock profile renews its keys, and pyarrow needs the
# endpoint spelled out
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AWS_PROFILE=flock
export AWS_ENDPOINT_URL=https://s3.flock.cerulean.works

echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), args: $*"
df -h "$(pwd)" | tail -1

srun uv run --frozen python -u -m src.recipe.cache --workers 32 "$@"
