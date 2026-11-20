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
# The 9-source macro bits per byte is then written to the training run's wandb summary as
# paloma_ppl (with paloma_ppl_step and paloma_ppl_model), which is what the OJ reads.
# The last eval to finish sets it, so score the checkpoint you mean to submit last.
#
# To score a checkpoint without touching what the OJ reads (e.g. intermediate checkpoints for
# an analysis), add --no-report as a fourth argument: the result stays in results.json only.
#
#   sbatch scripts/eval.sbatch dolma-v1_7-150m-30b 5600 8c5683d --no-report
#
# flock eval run wants a new or empty --out, so a rerun needs that directory moved away first
#SBATCH --job-name=eval
#SBATCH --account=ACD115198
#SBATCH --partition=h200_256gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

usage="usage: sbatch scripts/eval.sbatch <run-name> <step> <train-commit> [--no-report]"
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "$usage" >&2
    exit 2
fi

run=$1
step=$2
commit=$3
report=1
if [ $# -eq 4 ]; then
    if [ "$4" != "--no-report" ]; then
        echo "$usage" >&2
        exit 2
    fi
    report=0
fi
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

# flock from the project environment (uv.lock), the same flock.eval the training eval uses
uv run --frozen flock models import "$stage" "$model" --tag "$tag"
uv run --frozen flock models versions "$model"

uv run --frozen flock eval run \
    --model "flock:$model@$tag" \
    --suite paloma \
    --dtype float32 \
    --out "$out"

echo "results in $out/results.json"

if [ "$report" -eq 0 ]; then
    echo "--no-report: the training run's summary is left as it is"
    exit 0
fi

uv run --frozen python -m src.report_eval "$out/results.json" \
    --run-name "$run" --eval paloma --key paloma_ppl --step "$step" \
    --model "flock:$model@$tag"
