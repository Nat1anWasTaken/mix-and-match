#!/usr/bin/env bash
# Submit P0-1's source pipeline on Forerunner1, one Slurm chain per source, from the repo
# root on a login node, after fetch_hf_parquet.py has downloaded the Parquet subsets:
#
#   scripts/embedding_ml_transfer/submit_sources.sh            # every source
#   scripts/embedding_ml_transfer/submit_sources.sh spa_Latn   # just these
#
# Local sources (FineWeb, FineWeb-2 spa/deu/rus/fra), each step waiting on the last
# (sbatch --dependency=afterok), all offline on ct112 nodes:
#   1. parquet_to_lance.py   Parquet subset -> $MIRRORS/<source> (Lance)
#   2. prepare.py            Qwen3 tokens -> $STORES/<store>, --token-budget per source
#   3. glotlid               meta/glotlid/ (paragraph-level French and Chinese LID)
#   4. vban_count            meta/vban_count/ (V_ban tokens per document)
# The test splits only get step 1: eval_sets.py reads their text.
#
# Chinese (cmn_Hani) is already on Flock (s3://flock-datasets/fineweb-2), so it follows
# the repo's usual F1 path instead: prepare.py --stage fetch here mirrors the fragments
# it needs, then process, fineweb2_chinese and vban_count run as a chain.
#
# Stores are uploaded to s3://embedding-ml-transfer/sources/ by upload_sources.sh once
# their chains have finished. Every job logs to logs/<job name>-<job id>.out.

set -euo pipefail

cd "$(dirname "$(realpath "$0")")/../.."

RAW=/work1/$USER/hf-raw
MIRRORS=/work1/$USER/mirrors/embedding-ml-transfer
STORES=/work1/$USER/stores/embedding-ml-transfer
GLOTLID=/work1/$USER/models/lid/glotlid/model_v3.bin
TOKENIZER=Qwen/Qwen3-0.6B-Base

# source -> "picks record | token budget" (budgets sized for S2's 50B Multi-Train, plan
# P0-1; English also covers Mono-Train)
declare -A TRAIN=(
    [fineweb]="$RAW/fineweb/picks/sample--100BT.json|30000000000"
    [spa_Latn]="$RAW/fineweb-2/picks/data--spa_Latn--train.json|13000000000"
    [deu_Latn]="$RAW/fineweb-2/picks/data--deu_Latn--train.json|13000000000"
    [rus_Cyrl]="$RAW/fineweb-2/picks/data--rus_Cyrl--train.json|13000000000"
    [fra_Latn]="$RAW/fineweb-2/picks/data--fra_Latn--train.json|8000000000"
)
TESTS=(spa_Latn deu_Latn rus_Cyrl fra_Latn)
CHINESE_BUDGET=9000000000

mkdir -p logs "$MIRRORS" "$STORES"

# sbatch scripts/f1-cpu.sbatch with a job name and an optional dependency; prints the id
submit() {
    local name=$1 after=$2
    shift 2
    local dependency=()

    if [[ -n $after ]]; then
        dependency=(--dependency="afterok:$after")
    fi

    sbatch --parsable --job-name="$name" "${dependency[@]}" ${SBATCH_ARGS:-} \
        scripts/f1-cpu.sbatch "$@" | cut -d';' -f1
}

store_name() {
    if [[ $1 == fineweb ]]; then echo fineweb-qwen3; else echo "fineweb-2-$1-qwen3"; fi
}

submit_train() {
    local source=$1
    local picks=${TRAIN[$source]%%|*} budget=${TRAIN[$source]##*|}
    local mirror name store
    name=$(store_name "$source")
    mirror=$MIRRORS/${name%-qwen3}-train
    store=$STORES/$name

    local convert prepare lid vban
    convert=$(submit "convert-${name%-qwen3}" "" \
        scripts/embedding_ml_transfer/parquet_to_lance.py --picks "$picks" --output "$mirror")
    prepare=$(submit "prepare-$name" "$convert" prepare.py \
        --dataset "$mirror" --output "$store" --token-budget "$budget" \
        --tokenizer "$TOKENIZER" --run-name "prepare-$name" --no-wandb)
    lid=$(submit "score-glotlid-$name" "$prepare" -m src.recipe.scorers.glotlid \
        --store "$store" --model "$GLOTLID" --run-name "score-glotlid-$name" --no-wandb)
    vban=$(submit "score-vban_count-$name" "$lid" -m src.recipe.scorers.vban_count \
        --store "$store" --run-name "score-vban_count-$name")
    echo "$source: convert $convert -> prepare $prepare -> glotlid $lid -> vban $vban"
}

submit_test() {
    local language=$1
    local job
    job=$(submit "convert-fineweb-2-$language-test" "" \
        scripts/embedding_ml_transfer/parquet_to_lance.py \
        --picks "$RAW/fineweb-2/picks/data--$language--test.json" \
        --output "$MIRRORS/fineweb-2-$language-test")
    echo "$language test: convert $job"
}

submit_chinese() {
    local name=fineweb-2-cmn_Hani-qwen3
    local store=$STORES/$name mirror=$MIRRORS/fineweb-2-hani
    local arguments=(--dataset s3://flock-datasets/fineweb-2 --output "$store"
        --token-budget "$CHINESE_BUDGET" --tokenizer "$TOKENIZER"
        --run-name "prepare-$name" --no-wandb --mirror "$mirror")

    export AWS_CA_BUNDLE=${AWS_CA_BUNDLE:-/etc/pki/tls/certs/ca-bundle.crt}
    eval "$(flock s3 credentials --ttl 24h)"
    echo "== fetch $name (login node)"
    uv run --frozen python -u prepare.py "${arguments[@]}" --stage fetch

    local prepare tags vban
    prepare=$(submit "prepare-$name" "" prepare.py "${arguments[@]}" --stage process)
    tags=$(submit "score-fineweb2_chinese-$name" "$prepare" \
        -m src.recipe.scorers.fineweb2_chinese --store "$store" --mirror "$mirror" \
        --run-name "score-fineweb2_chinese-$name" --no-wandb --stage process)
    vban=$(submit "score-vban_count-$name" "$tags" -m src.recipe.scorers.vban_count \
        --store "$store" --run-name "score-vban_count-$name")
    echo "cmn_Hani: prepare $prepare -> fineweb2_chinese $tags -> vban $vban"
}

selected=("$@")

if [[ ${#selected[@]} -eq 0 ]]; then
    selected=("${!TRAIN[@]}" test cmn_Hani)
fi

for source in "${selected[@]}"; do
    case $source in
        test) for language in "${TESTS[@]}"; do submit_test "$language"; done ;;
        cmn_Hani) submit_chinese ;;
        *) submit_train "$source" ;;
    esac
done
