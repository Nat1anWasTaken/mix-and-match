#!/usr/bin/env bash
# Run a data script on Forerunner1, whose compute nodes have no internet. From the repo
# root on a login node, with the script's arguments (the same for every stage):
#
#   scripts/submit-f1.sh prepare.py \
#       --dataset s3://flock-datasets/fineweb-edu --mirror /work1/$USER/mirrors/fineweb-edu \
#       --output /work1/$USER/stores/fineweb-edu-llama3 \
#       --remote s3://<bucket>/sources/fineweb-edu-llama3 \
#       --token-budget 30000000000 --run-name prepare-fineweb-edu-llama3
#
#   scripts/submit-f1.sh -m src.recipe.scorers.paloma_decontamination \
#       --store /work1/$USER/stores/fineweb-edu-llama3 \
#       --mirror /work1/$USER/mirrors/fineweb-edu \
#       --remote s3://<bucket>/sources/fineweb-edu-llama3 \
#       --run-name score-paloma_decontamination-fineweb-edu-llama3
#
# It runs:
#   1. --stage fetch here, which downloads into --mirror and the Hugging Face cache
#   2. sbatch scripts/f1-cpu.sbatch ... --stage process, offline on a compute node
#   3. a background watcher (log: logs/submit-<job>.log) that syncs the wandb run live
#      while the job runs, then runs --stage upload here if the job completed. Without
#      --remote it skips the upload; rerun with --stage upload by hand any time
#
# Extra sbatch options go in SBATCH_ARGS, e.g. SBATCH_ARGS="--partition=development
# --time=08:00:00".

set -euo pipefail

self=$(realpath "$0")
cd "$(dirname "$self")/.."

# pyarrow's S3 client needs RHEL's CA bundle (see open_filesystem in src/tokenizer.py)
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/pki/tls/certs/ca-bundle.crt}"

# The value of option $1 in the remaining arguments, or nothing
option() {
    local name=$1
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            "$name") echo "${2:-}"; return ;;
            "$name"=*) echo "${1#*=}"; return ;;
        esac
        shift
    done
}

has_flag() {
    local name=$1
    shift

    for argument in "$@"; do
        [[ $argument == "$name" ]] && return 0
    done

    return 1
}

s3_credentials() {
    eval "$(flock s3 credentials --ttl 24h)"
}

job_running() {
    [[ -n $(squeue -h -j "$1" 2>/dev/null) ]]
}

# --- watcher: re-run of this file in the background, see the end ---------------------

if [[ ${1:-} == --watch ]]; then
    jobid=$2
    wandb_dir=$3
    run_name=$4
    shift 4

    echo "watching job $jobid"

    sync_pid=""

    if [[ -n $run_name ]]; then
        # The job's run directory appears once wandb.init runs in it. wandb_dir is this
        # job's own (see f1-cpu.sbatch), so a run left by an earlier job of the same
        # name is never picked up. Checked often, so a short job is still synced while
        # it runs; once more after the job ends, in case it finished between checks
        run_dir=""

        while true; do
            run_dir=$(ls -d "$wandb_dir"/wandb/offline-run-* 2>/dev/null | head -1 || true)

            if [[ -n $run_dir ]] || ! job_running "$jobid"; then
                break
            fi

            sleep 5
        done

        if [[ -z $run_dir ]]; then
            run_dir=$(ls -d "$wandb_dir"/wandb/offline-run-* 2>/dev/null | head -1 || true)
        fi

        if [[ -n $run_dir ]]; then
            echo "syncing $run_dir to wandb run $run_name"
            # --id: a resumed run writes a new directory, which joins the same wandb run
            uv run --frozen wandb beta sync --live --yes --id "$run_name" "$run_dir" &
            sync_pid=$!
        fi
    fi

    while job_running "$jobid"; do
        sleep 60
    done

    if [[ -n $sync_pid ]]; then
        # --live ends by itself once the run finished cleanly, but can hang after a
        # crash, so give it a few minutes
        for _ in $(seq 30); do
            kill -0 "$sync_pid" 2>/dev/null || break
            sleep 10
        done

        if kill -0 "$sync_pid" 2>/dev/null; then
            echo "wandb sync still running after the job ended; stopping it"
            kill "$sync_pid"
        fi

        wait "$sync_pid" || true
    fi

    state=$(sacct -j "$jobid" -X -n -P -o State | head -1)
    echo "job $jobid ended: $state"

    if [[ $state != COMPLETED ]]; then
        echo "not uploading; fix the job and resubmit (finished work is kept)"
        exit 1
    fi

    if [[ -z $(option --remote "$@") ]]; then
        echo "no --remote, not uploading"
        exit 0
    fi

    # The key from submission may have expired by now
    s3_credentials
    uv run --frozen python -u "$@" --stage upload
    exit 0
fi

# --- submit ----------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    sed -n '2,/^$/s/^# \{0,1\}//p' "$self"
    exit 2
fi

if [[ -n $(option --stage "$@") ]]; then
    echo "don't pass --stage; this script runs fetch, process and upload itself" >&2
    exit 2
fi

if [[ $1 == -m ]]; then
    script_name=${2##*.}
else
    script_name=$(basename "$1" .py)
fi

run_name=$(option --run-name "$@")

if has_flag --no-wandb "$@"; then
    run_name=""
fi

job_name=${run_name:-$script_name}
# Each job writes its run under its own <job id>/ in here (see f1-cpu.sbatch)
wandb_root=$PWD/logs/wandb/$job_name
mkdir -p logs "$wandb_root"

uv sync --frozen
s3_credentials

echo "== fetch"
uv run --frozen python -u "$@" --stage fetch

echo "== process"
# shellcheck disable=SC2086  # SBATCH_ARGS is a list of options
jobid=$(
    F1_WANDB_ROOT=$wandb_root sbatch --parsable --job-name="$job_name" ${SBATCH_ARGS:-} \
        scripts/f1-cpu.sbatch "$@" --stage process
)
jobid=${jobid%%;*}
echo "submitted job $jobid, log: logs/$job_name-$jobid.out"

# nohup + setsid: the watcher outlives this shell and the terminal
setsid nohup "$self" --watch "$jobid" "$wandb_root/$jobid" "$run_name" "$@" \
    > "logs/submit-$jobid.log" 2>&1 < /dev/null &

echo "watcher log: logs/submit-$jobid.log (wandb sync, then upload)"
