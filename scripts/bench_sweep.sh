#!/bin/bash
# Sweep n_cpu_moe / context / KV type with llama-bench.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-$ROOT/models/gpt-oss-120b-MXFP4.gguf}"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
RUNNER="$ROOT/scripts/run_with_vram.py"
NCMOE_LIST="${NCMOE_LIST:-4 8 10 12 14 16 20}"
CTX_LIST="${CTX_LIST:-8192}"
KV_LIST="${KV_LIST:-f16}"
PROMPT="${PROMPT:-512}"
GEN="${GEN:-128}"
REPS="${REPS:-2}"
OUT="${OUT:-$ROOT/measurements/sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "model=$MODEL" | tee "$OUT/README.txt"
echo "out=$OUT" | tee -a "$OUT/README.txt"

for ctx in $CTX_LIST; do
  for kv in $KV_LIST; do
    for n in $NCMOE_LIST; do
      name="ctx${ctx}_kv${kv}_ncmoe${n}"
      echo "=== $name ==="
      python3 "$RUNNER" --metrics "$OUT/$name.metrics.json" -- \
        "$BENCH" -m "$MODEL" -ngl 99 -ncmoe "$n" \
        -p "$PROMPT" -n "$GEN" -d "$ctx" -r "$REPS" \
        -ctk "$kv" -ctv "$kv" -fa auto -o json \
        > "$OUT/$name.json" 2> "$OUT/$name.log"
      rc=$?
      echo "rc=$rc"
      if [ $rc -ne 0 ]; then
        echo "$name rc=$rc" >> "$OUT/failures.txt"
      fi
    done
  done
done
echo "done: $OUT"
