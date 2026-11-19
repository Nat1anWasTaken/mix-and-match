#!/usr/bin/env bash
# Train on one H200 by default. All arguments go straight to train.py:
#
#   mkdir -p logs
#   sbatch --job-name=150M-baseline scripts/train.sbatch \
#       --recipe s3://mix-and-match/recipes/dolma-v1_7-30b --model-size 150M \
#       --total-tokens 30001856512 --eval paloma@val --run-name 150M-baseline
#
# --eval is required: the flock suite or task scored during training, by its registry
# name, optionally on another split (paloma@val; plain paloma is the test split).
#
# --total-tokens must be a multiple of --global-batch-tokens (4,194,304 by default),
# so "30B" is 7153 steps = 30,001,856,512 tokens.
#
# For more GPUs on the node, override the resources on the sbatch line. torchrun
# starts one rank per allocated GPU, so nothing else changes:
#
#   sbatch --gpus-per-node=H200:8 --cpus-per-task=96 --mem=1000G --time=04:45:00 ...
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
# pyarrow takes the keys from the profile but ignores its endpoint_url, and without
# an endpoint it asks AWS for the bucket ("Bucket ... not found")
export AWS_ENDPOINT_URL=https://s3.flock.cerulean.works

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, commit $(git rev-parse --short HEAD)"
echo "args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# One task per node; torchrun forks a rank per GPU and sets RANK/LOCAL_RANK/WORLD_SIZE,
# which train.py's setup_distributed reads. With one GPU it runs a single rank
export PYTHONUNBUFFERED=1
# NCCL's bootstrap otherwise picks the IPoIB interface (ibp*), where connecting to
# the node's own address never completes, so init hangs with idle GPUs. --standalone
# keeps every rank on this node, so loopback is enough
export NCCL_SOCKET_IFNAME=lo
srun uv run --frozen torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    train.py "$@"
