#!/usr/bin/env bash
# Upload the offline wandb runs of P0-1's Forerunner1 jobs, from the repo root on a
# login node (compute nodes have no internet, so f1-cpu.sbatch runs wandb offline):
#
#   scripts/embedding_ml_transfer/sync_wandb.sh
#
# Each job wrote its run under logs/wandb/embedding-ml-transfer/<job id>/wandb/. A run
# whose job has ended is synced once, then marked with a .synced file; runs of jobs
# still in the queue are left for the next call. The run id is the run name (see
# src/loggers.py), so a rerun of the same name joins its earlier wandb run.

set -euo pipefail

cd "$(dirname "$(realpath "$0")")/../.."

ROOT=logs/wandb/embedding-ml-transfer

for job in "$ROOT"/*/; do
    job=${job%/}
    id=$(basename "$job")

    if [[ -n $(squeue -h -j "$id" 2>/dev/null) ]]; then
        echo "job $id still running, skipped"
        continue
    fi

    for run in "$job"/wandb/offline-run-*; do
        [[ -d $run && ! -e $run/.synced ]] || continue
        echo "== sync $run"
        uv run --frozen wandb sync "$run" && touch "$run/.synced"
    done
done
