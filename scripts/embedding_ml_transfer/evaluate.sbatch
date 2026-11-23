#!/usr/bin/env bash
# One H200 for evaluate.py / bli.py / compare.py; arguments are the command to run:
#
#   sbatch scripts/embedding_ml_transfer/evaluate.sbatch \
#       scripts/embedding_ml_transfer/evaluate.py --checkpoint ... --output ...
#
# Run it from the worktree whose code should score (the job cds to the submit dir).
#SBATCH --job-name=eval-emt
#SBATCH --account=ACD115198
#SBATCH --partition=h200_256gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --exclude=25a-hgpn[069,141,173-176]
#SBATCH --output=/work/u9216411/acd115198/embedding-ml-transfer-runs/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda/13.0
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AWS_PROFILE=flock AWS_ENDPOINT_URL=https://s3.flock.cerulean.works

echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD)"
echo "args: $*"
srun uv run --frozen python "$@"
