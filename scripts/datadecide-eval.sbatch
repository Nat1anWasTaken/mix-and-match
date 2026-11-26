#!/usr/bin/env bash
# Score DataDecide checkpoints on Paloma (float32, full test split), one manifest row per
# array task:
#
#   sbatch --array=0-74%8 --job-name=eval-datadecide-tier1 \
#       scripts/datadecide-eval.sbatch src/datadecide/manifest.csv 1
#
# The second argument is the manifest's tier; array index i is that tier's i-th row (0-based,
# in file order). %8 keeps at most 8 GPUs busy at once.
#
# Each task converts its checkpoint (src/datadecide/convert.py), scores the folder as
# <folder>@<upstream commit> so results.json names the exact weights, then deletes the
# converted model and its download: a 1B checkpoint is about 4.7 GB twice over.
# Results stay in $WORK/datadecide/results/<name>/results.json. A task whose results.json
# already exists does nothing, so resubmitting the same array only redoes what failed.
#
# Each result is also logged to Flock's lab7-data-curation project as an eval run named
# datadecide-paloma-<name>. Its summary holds only flock's <task>/<metric> and suites/<suite>
# keys, not paloma_ppl or the token counts the OJ reads.
#
# --batch-tokens is 131072, four times flock's default, as the default left an H200 at about
# 10 GB of 141. A 1B model's logits for one batch are then 131072 x 50304 FP32, about 26 GB.
# Batching moves FP32 scores very slightly (flock's docs: per-domain bits per byte within
# about 1e-7), so score every tier with the same value.
#SBATCH --job-name=eval-datadecide
#SBATCH --account=ACD115198
#SBATCH --partition=h200_256gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ $# -ne 2 ]; then
    echo "usage: sbatch --array=0-<n-1> scripts/datadecide-eval.sbatch <manifest.csv> <tier>" >&2
    exit 2
fi
manifest=$1
tier=$2

# repo, revision and name of this task's row
read -r repo revision name < <(python3 -I - "$manifest" "$tier" "$SLURM_ARRAY_TASK_ID" <<'EOF'
import csv, sys
path, tier, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [r for r in csv.DictReader(open(path)) if r["tier"] == tier]
row = rows[index]
print(row["repo"], row["revision"], row["name"])
EOF
)

root=$WORK/datadecide
model=$root/models/$name
out=$root/results/$name
# this task's downloads only; HF_HOME stays as it is, for the Hub token and the Paloma cache
cache=$root/hf-cache/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID

echo "job $SLURM_JOB_ID ($SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]) on $(hostname), GPU $CUDA_VISIBLE_DEVICES"
echo "commit $(git rev-parse --short HEAD), $repo@$revision -> $out"

if [ -f "$out/results.json" ]; then
    echo "$out/results.json exists, nothing to do"
    exit 0
fi
# a task that died mid-way leaves these; flock eval run wants a new or empty --out
rm -rf "$model" "$out"
trap 'rm -rf "$model" "$cache"' EXIT

uv run --frozen python -m src.datadecide.convert \
    --repo "$repo" --revision "$revision" --out "$model" --cache-dir "$cache"
rm -rf "$cache"
commit=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["commit"])' "$model/datadecide.json")

mkdir -p "$root/results"
# the run name; flock's tracker logging takes it from the wandb SDK's environment
export WANDB_NAME=datadecide-paloma-$name
uv run --frozen flock eval run \
    --model "$model@$commit" \
    --suite paloma \
    --dtype float32 \
    --batch-tokens 131072 \
    --wandb-project lab7-data-curation \
    --out "$out"
cp "$model/datadecide.json" "$out/"

echo "results in $out/results.json"
