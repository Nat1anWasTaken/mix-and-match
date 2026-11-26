#!/usr/bin/env bash
# Convert one DataDecide checkpoint and check it against hf_olmo (src/datadecide/parity.py):
#
#   sbatch --job-name=parity-datadecide-dolma1_7-20M scripts/datadecide-parity.sbatch \
#       allenai/DataDecide-dolma1_7-20M step14594-seed-default
#
# Files go under $WORK/datadecide/: models/<name>/ (the converted model, kept), parity/<name>.pt
# (hf_olmo's scores) and hf-cache/parity/ (the conversion's download). The job fails if the converted model's
# log-probabilities differ from hf_olmo's by more than 1e-3.
#SBATCH --job-name=parity-datadecide
#SBATCH --account=ACD115198
#SBATCH --partition=h200_256gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ $# -ne 2 ]; then
    echo "usage: sbatch scripts/datadecide-parity.sbatch <repo> <revision>" >&2
    exit 2
fi
repo=$1
revision=$2
name=${repo#allenai/DataDecide-}-$revision
root=$WORK/datadecide

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES, commit $(git rev-parse --short HEAD)"

# convert.py refuses an existing folder; this one is only ever this job's output
rm -rf "$root/models/$name"
uv run --frozen python -m src.datadecide.convert \
    --repo "$repo" --revision "$revision" --out "$root/models/$name" --cache-dir "$root/hf-cache/parity"

# ai2-olmo imports datasets without depending on it, and needs numpy<2, which today's pyarrow
# refuses: resolve everything as of 2025-04-01, with transformers as the checkpoints were saved
uv run --isolated --no-project --python 3.12 --exclude-newer 2025-04-01 \
    --with ai2-olmo==0.6.0 --with datasets --with transformers==4.50.3 \
    python src/datadecide/parity.py reference \
    --repo "$repo" --revision "$revision" --out "$root/parity/$name.pt"

uv run --frozen python -m src.datadecide.parity compare \
    --model "$root/models/$name" --reference "$root/parity/$name.pt"
