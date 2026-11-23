#!/usr/bin/env bash
# Copy a recipe and its shards to /work for train.py --recipe-cache. Reading straight
# from S3, every rank reads the whole shard stream and 8 ranks hold t1-l12 at ~0.45M
# tok/s; from the cache t1-l24 runs at ~1M tok/s on 8 H200 (P0-2a run log).
#
#   sbatch scripts/embedding_ml_transfer/cache.sbatch s3://embedding-ml-transfer/recipes/<r> ...
#SBATCH --job-name=cache-emt
#SBATCH --account=ACD115198
#SBATCH --partition=h200_256gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn[069,141,173-176]
#SBATCH --output=/work/u9216411/acd115198/embedding-ml-transfer-runs/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AWS_PROFILE=flock AWS_ENDPOINT_URL=https://s3.flock.cerulean.works
for recipe in "$@"; do
    srun uv run --frozen python -m src.recipe.cache --recipe "$recipe" \
        --cache /work/u9216411/acd115198/recipe-cache
done
