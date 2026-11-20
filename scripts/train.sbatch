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
# For several nodes, also pass --nodes. h200_256gpu is our project's own partition: it
# takes any GPU count up to the account's 256 (shared with the team, so check
# `squeue -a -p h200_256gpu` first) and has no wall-time limit. --mem and
# --cpus-per-task are per node. One torchrun starts on each node and they meet at the
# first node:
#
#   sbatch --nodes=4 --gpus-per-node=H200:8 --cpus-per-task=96 \
#       --mem=1000G --time=... scripts/train.sbatch ...
#
# H200-hours = GPUs x wall time, so --time is what bounds the grant a run can use.
# S3 keys come from the flock AWS profile, which renews them, so --time is not bound
# by a key's lifetime. Set it up once on the cluster with `flock s3 configure`
#SBATCH --job-name=train
#SBATCH --account=ACD115198
#SBATCH --partition=h200_256gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

if [[ "$SLURM_NNODES" -eq 1 ]]; then
    # NCCL's bootstrap otherwise picks the IPoIB interface (ibp*), where connecting to
    # the node's own address never completes, so init hangs with idle GPUs. --standalone
    # keeps every rank on this node, so loopback is enough
    export NCCL_SOCKET_IFNAME=lo
    srun uv run --frozen torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" \
        train.py "$@"
    exit
fi

# Several nodes: the ranks meet at the first node's torchrun (static rendezvous).
# Its hostname resolves to the node's vlan1721 address, so the TCPStore and NCCL's
# bootstrap both use vlan1721 (Ethernet) and never touch IPoIB
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))
export MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME=vlan1721
# Gradients go over the 8 InfiniBand NDR HCAs, one PCIe-local to each GPU. mlx5_1 and
# mlx5_2 are the RoCE ports under bond0, which NCCL would otherwise also list. The
# leading = makes the names exact matches rather than prefixes
export NCCL_IB_HCA==mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
# Print the chosen transports once; check for "via NET/IB/.../GDRDMA" in the log
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
echo "nodes $SLURM_JOB_NODELIST, master $MASTER_ADDR:$MASTER_PORT"

# The single quotes defer $SLURM_NODEID and $SLURM_GPUS_ON_NODE to each node's task;
# expanded here they would be the first node's values on every node. train.py's
# arguments follow as "$@" after the placeholder $0 (_). --kill-on-bad-exit stops the
# whole job when one node fails instead of leaving the others blocked in a collective.
# --no-sync keeps the nodes from syncing the shared .venv at the same time
srun --kill-on-bad-exit=1 bash -c 'exec uv run --frozen --no-sync torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train.py "$@"' _ "$@"
