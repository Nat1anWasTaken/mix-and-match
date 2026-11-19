#!/usr/bin/env bash
# Import a run's checkpoint into Flock and score it on Paloma (float32, full test split):
#
#   sbatch --job-name=eval-paloma-<run-name> scripts/eval.sbatch <run-name> <step> <train-commit>
#
# Chain it to a training job so it starts once training succeeds:
#
#   sbatch --dependency=afterok:<train job id> --job-name=eval-paloma-dolma-v1_7-150m-30b \
#       scripts/eval.sbatch dolma-v1_7-150m-30b 7153 8c5683d
#
# <train-commit> is the commit the training log's first line prints. Pass it in rather than
# reading HEAD here, since the checkout may have moved on by the time this job starts.
#
# The checkpoint becomes a version of cerulean-labs/<run-name> tagged
# <run-name>-step<step>-<commit>, which names the run, step and code it came from, so the tag
# stays unique across every model we push. Results go to checkpoints/<run-name>/paloma-step<step>/.
# flock eval run wants a new or empty --out, so a rerun needs that directory moved away first
#SBATCH --job-name=eval
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ $# -ne 3 ]; then
    echo "usage: sbatch scripts/eval.sbatch <run-name> <step> <train-commit>" >&2
    exit 2
fi

run=$1
step=$2
commit=$3
tag=$run-step$step-$commit
checkpoint=checkpoints/$run/step-$step
model=cerulean-labs/$run
out=checkpoints/$run/paloma-step$step

echo "job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES"
echo "checkpoint $checkpoint -> flock:$model@$tag"
test -f "$checkpoint/config.json"

# A checkpoint also holds resume.pt and trainer_state.json for resuming the run. Import only
# the HF files (config, safetensors, tokenizer), hard-linked so nothing is copied
stage=checkpoints/$run/flock-import-step$step
rm -rf "$stage"
mkdir -p "$stage"
for file in "$checkpoint"/*; do
    case "$(basename "$file")" in
        resume.pt | trainer_state.json) ;;
        *) ln "$file" "$stage/" ;;
    esac
done
ls -la "$stage"

flock models import "$stage" "$model" --tag "$tag"
flock models versions "$model"

flock eval run \
    --model "flock:$model@$tag" \
    --suite paloma \
    --dtype float32 \
    --out "$out"

echo "results in $out/results.json"
