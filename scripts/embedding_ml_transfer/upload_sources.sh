#!/usr/bin/env bash
# Upload P0-1's finished stores and eval text to s3://embedding-ml-transfer, from the
# repo root on a login node (only login nodes reach S3):
#
#   scripts/embedding_ml_transfer/upload_sources.sh                 # every store
#   scripts/embedding_ml_transfer/upload_sources.sh fake-fra-qwen3  # just these
#
# A store is uploaded only once prepare.py finished it (manifest.json) and every scorer
# column it needs is complete, so the compilers never read a half-scored store. The
# upload skips files already there with the same size, so rerun it any time, e.g.
# after adding a meta/ column.

set -euo pipefail

cd "$(dirname "$(realpath "$0")")/../.."

STORES=/work1/$USER/stores/embedding-ml-transfer
EVAL=/work1/$USER/p0-1/eval
BUCKET=s3://embedding-ml-transfer

# store -> the meta/ kinds it must have
declare -A KINDS=(
    [fineweb-qwen3]="glotlid glotlid_v2 vban_count"
    [fineweb-2-spa_Latn-qwen3]="glotlid glotlid_v2 vban_count"
    [fineweb-2-deu_Latn-qwen3]="glotlid glotlid_v2 vban_count"
    [fineweb-2-rus_Cyrl-qwen3]="glotlid glotlid_v2 vban_count"
    [fineweb-2-fra_Latn-qwen3]="glotlid vban_count"
    [fineweb-2-cmn_Hani-qwen3]="fineweb2_chinese vban_count"
    [fake-fra-qwen3]=""
)

export AWS_CA_BUNDLE=${AWS_CA_BUNDLE:-/etc/pki/tls/certs/ca-bundle.crt}
eval "$(flock s3 credentials --ttl 24h)"

selected=("$@")

if [[ ${#selected[@]} -eq 0 ]]; then
    selected=("${!KINDS[@]}")
fi

for name in "${selected[@]}"; do
    store=$STORES/$name

    if [[ ! -f $store/manifest.json ]]; then
        echo "skip $name: not prepared yet"
        continue
    fi

    # shellcheck disable=SC2086  # KINDS values are lists
    uv run --frozen python -I - "$store" ${KINDS[$name]} <<'EOF' || continue
import sys
sys.path.insert(0, ".")
from src.recipe.metadata import is_complete
from src.tokenizer import ShardStore

store = ShardStore(sys.argv[1])
missing = [kind for kind in sys.argv[2:] if not is_complete(store, kind)]

if missing:
    print(f"skip {sys.argv[1]}: meta/{', meta/'.join(missing)} not complete")
    sys.exit(1)
EOF

    echo "== upload $name"
    uv run --frozen python -I -c "
import sys; sys.path.insert(0, '.')
from src.stages import upload_store
upload_store('$store', '$BUCKET/sources/$name')"
done

if [[ -d $EVAL ]]; then
    echo "== upload eval text"
    uv run --frozen python -I -c "
import sys; sys.path.insert(0, '.')
from src.stages import upload_store
upload_store('$EVAL', '$BUCKET/eval')"
fi
