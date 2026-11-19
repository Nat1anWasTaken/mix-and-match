#!/usr/bin/env bash
# Train on one H200. All arguments go straight to train.py:
#
#   mkdir -p logs
#   sbatch --job-name=150M-baseline scripts/train.sbatch \
#       --recipe s3://<bucket>/recipes/fwedu-identity-30b --model-size 150M \
#       --total-tokens 30000000000 --run-name 150M-baseline --compile
#
# H200-hours = GPUs x wall time, so --time is what bounds the grant a run can use.
# S3 keys come from the flock AWS profile, which renews them, so --time is not bound
# by a key's lifetime. Set it up once on the cluster with `flock s3 configure`
#SBATCH --job-name=train
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=07:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load cuda/13.0

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4
export WANDB_MODE="${WANDB_MODE:-online}"

# The data loader reads shards from S3 while training. A key from `flock s3
# credentials` in AWS_* variables would expire mid-run, so use the flock profile
# instead: its credential_process lets pyarrow's S3 client fetch a new key when the old
# one expires. The AWS SDK prefers keys in the environment, so none may be set
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AWS_PROFILE=flock

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, commit $(git rev-parse --short HEAD)"
echo "args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun uv run --frozen python -u train.py "$@"
