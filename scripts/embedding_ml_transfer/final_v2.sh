#!/usr/bin/env bash
# P0-1 final pass: upload glotlid_v2, compile the v2 recipes (all lines, P > 0.8),
# check their set relations, run the final audit with a fresh seed (43)
set -euo pipefail
cd /work1/nat1andotxyz/mix-and-match-embedding-ml-transfer
export AWS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
eval "$(flock s3 credentials --ttl 24h)"
scripts/embedding_ml_transfer/upload_sources.sh fineweb-qwen3 fineweb-2-spa_Latn-qwen3 fineweb-2-deu_Latn-qwen3 fineweb-2-rus_Cyrl-qwen3 2>&1 | grep -E "^==|uploaded [0-9]+ of [0-9]+ files|skip"
B=s3://embedding-ml-transfer
C="uv run --frozen python -m src.recipe.compilers.embedding_ml_transfer --store-root $B/sources --lid-kind glotlid_v2 --line-length all --line-threshold 0.8 --workers 32"
$C --split multi_train --total-tokens 20000000000 --output $B/recipes/multi-train-20b-v2 > logs/compile-multi-train-20b-v2.log 2>&1
$C --split mono_train --total-tokens 20000000000 --output $B/recipes/mono-train-20b-v2 > logs/compile-mono-train-20b-v2.log 2>&1
$C --split oracle_train --total-tokens 24000000000 --output $B/recipes/oracle-train-24b-v2 > logs/compile-oracle-train-24b-v2.log 2>&1
$C --split zh_adapt --total-tokens 2000000000 --output $B/recipes/zh-adapt-2b-v2 > logs/compile-zh-adapt-2b-v2.log 2>&1
$C --split fake_fra_adapt --total-tokens 2000000000 --output $B/recipes/fake-fra-adapt-2b-v2 > logs/compile-fake-fra-adapt-2b-v2.log 2>&1
for f in logs/compile-*-v2.log; do echo "== $f"; grep -E "^(en|es|de|ru|zh):|wrote" $f; done
uv run --frozen python -I - <<'PY'
import sys, glob, os; sys.path.insert(0,".")
import numpy as np
from src.stages import upload_store
L="/work1/nat1andotxyz/p0-1/recipes"
names=["multi-train-20b-v2","mono-train-20b-v2","oracle-train-24b-v2","zh-adapt-2b-v2","fake-fra-adapt-2b-v2"]
for r in names:
    if not os.path.exists(f"{L}/{r}/recipe.json"): upload_store(f"s3://embedding-ml-transfer/recipes/{r}", f"{L}/{r}")
def sel(r,c): return {os.path.basename(f):np.load(f) for f in glob.glob(f"{L}/{r}/selections/main/{c}/*.npy")}
ok=True
for c in ["en","es","de","ru"]:
    a,b=sel("multi-train-20b-v2",c),sel("oracle-train-24b-v2",c); same=a.keys()==b.keys() and all(np.array_equal(v,b[k]) for k,v in a.items()); ok&=same; print(f"multi/{c} == oracle/{c}:", same)
m,o=sel("multi-train-20b-v2","en"),sel("mono-train-20b-v2","en"); s=all(np.isin(v,o.get(k,np.array([],np.uint32))).all() for k,v in m.items()); ok&=s; print("multi/en ⊂ mono/en:", s)
for a,b,c in [("zh-adapt-2b-v2","oracle-train-24b-v2","zh"),("fake-fra-adapt-2b-v2","oracle-train-24b-v2","fake_fra")]:
    x,y=sel(a,c),sel(b,c); d=not any(np.isin(v,y.get(k,np.array([],np.uint32))).any() for k,v in x.items()); ok&=d; print(f"{a} ∩ oracle/{c} = ∅:", d)
open("/work1/nat1andotxyz/p0-1/relations_ok","w").write(str(ok))
PY
export F1_WANDB_ROOT=$PWD/logs/wandb/embedding-ml-transfer
L=/work1/nat1andotxyz/p0-1/recipes
j=$(sbatch --parsable --job-name=audit-p0-1-v2 --time=02:00:00 scripts/f1-cpu.sbatch scripts/embedding_ml_transfer/audit.py --recipe multi_train=$L/multi-train-20b-v2 --recipe mono_train=$L/mono-train-20b-v2 --recipe oracle_train=$L/oracle-train-24b-v2 --recipe zh_adapt=$L/zh-adapt-2b-v2 --recipe fake_fra_adapt=$L/fake-fra-adapt-2b-v2 --store-map s3://embedding-ml-transfer/sources=/work1/nat1andotxyz/stores/embedding-ml-transfer --lid-model /work1/nat1andotxyz/models/lid/lid.176.bin --eval-text zh_fineweb2_test=/work1/nat1andotxyz/p0-1/eval/fineweb-2-cmn_Hani-test.jsonl --seed 43 --output /work1/nat1andotxyz/p0-1/audit-v2 --run-name audit-p0-1-v2 --wandb-project embedding-ml-transfer)
echo "audit job $j"
while squeue -h -j $j 2>/dev/null | grep -q .; do sleep 30; done
sacct -j $j -X -o JobID,State,Elapsed
cat /work1/nat1andotxyz/p0-1/audit-v2/audit.md
