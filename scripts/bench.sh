#!/usr/bin/env bash
# bench.sh — one command, one table: reproduce the decode/prefill numbers on
# this box using the auto-fit placement.
#
# For every "ctx:kv:streams" workload it:
#   1. asks scripts/moe_fit.py for the placement (ncmoe, op_offload),
#   2. starts llama-server through scripts/bench_serve.py,
#   3. prints decode tok/s, prefill tok/s and peak VRAM.
#
# Usage:
#   ./scripts/bench.sh
#   CONFIGS="8192:f16:1 32768:f16:1" REPS=5 ./scripts/bench.sh
#
# Environment:
#   MODEL          GGUF path      (default models/gpt-oss-120b-MXFP4.gguf)
#   LLAMA_SERVER   llama-server   (default build/bin/...)
#   CONFIGS        ctx:kv:streams list (default "8192:f16:1 32768:q8_0:1 8192:f16:2")
#   REPS           timed runs per config (default 3, after 1 warmup)
#   N_PREDICT      decode tokens per request (default 128)
#   PROMPT_REPS    repeat the built-in prompt N times (default 1)
#   FORCE=1        run even when moe_fit reports the host would thrash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$ROOT/models/gpt-oss-120b-MXFP4.gguf}"
BIN="${LLAMA_SERVER:-$ROOT/build/bin/llama-server}"
PYTHON="${PYTHON:-python3}"
CONFIGS="${CONFIGS:-8192:f16:1 32768:q8_0:1 8192:f16:2}"
REPS="${REPS:-3}"
WARMUP="${WARMUP:-1}"
N_PREDICT="${N_PREDICT:-128}"
PROMPT_REPS="${PROMPT_REPS:-1}"
FORCE="${FORCE:-0}"
OUT="${OUT:-$ROOT/results/bench_$(date +%Y%m%d_%H%M%S)}"

die() { echo "bench: $*" >&2; exit 1; }

[ -f "$MODEL" ] || die "model not found: $MODEL (run scripts/download_modelscope.py or set MODEL=)"
[ -x "$BIN" ] || die "llama-server not found: $BIN (build it, or set LLAMA_SERVER=)"
[ -f "$ROOT/scripts/moe_fit.py" ] || die "missing scripts/moe_fit.py"
[ -f "$ROOT/scripts/bench_serve.py" ] || die "missing scripts/bench_serve.py"

mkdir -p "$OUT"
echo "bench: model  $MODEL"
echo "bench: server $BIN"
echo "bench: configs $CONFIGS"
echo "bench: output $OUT"
echo

rows="$OUT/rows.tsv"
: > "$rows"
failed=0
ran=0

for cfg in $CONFIGS; do
  ctx="${cfg%%:*}"
  rest="${cfg#*:}"
  kv="${rest%%:*}"
  streams="${rest#*:}"

  case "$ctx" in ''|*[!0-9]*) die "bad config '$cfg': ctx must be an integer";; esac
  case "$streams" in ''|*[!0-9]*) die "bad config '$cfg': streams must be an integer";; esac
  [ -n "$kv" ] || die "bad config '$cfg': expected ctx:kv:streams"

  name="ctx${ctx}_kv${kv}_x${streams}"
  label="$("$PYTHON" -I -c 'import sys; print(f"{int(sys.argv[1]) // 1024}k")' "$ctx")/${kv} x${streams}"

  echo "=== $name ==="
  plan_json="$("$PYTHON" -I "$ROOT/scripts/moe_fit.py" --json \
      --model "$MODEL" --bin "$BIN" --ctx "$ctx" --kv-dtype "$kv" \
      2> "$OUT/$name.fit.err")"
  if [ $? -ne 0 ] || [ -z "$plan_json" ]; then
    echo "  placement failed: $(tail -n 1 "$OUT/$name.fit.err" 2>/dev/null)"
    failed=$((failed + 1))
    continue
  fi

  read -r ncmoe op_offload host_ok < <(printf '%s' "$plan_json" | "$PYTHON" -I -c '
import json, sys
p = json.load(sys.stdin)
print(p["ncmoe"], p["op_offload"], 1 if p["host_ok"] else 0)
')
  printf '%s' "$plan_json" | "$PYTHON" -I -c '
import json, sys
for n in json.load(sys.stdin).get("notes", []):
    print("  note:", n)
'
  echo "  placement: ncmoe=$ncmoe op_offload=$op_offload"

  if [ "$host_ok" -eq 0 ] && [ "$FORCE" != "1" ]; then
    echo "  skipping: moe_fit says the CPU-resident experts do not fit in available host RAM"
    echo "  (set FORCE=1 to run anyway; expect page eviction and single-digit tok/s)"
    failed=$((failed + 1))
    continue
  fi

  args=(--bin "$BIN" --model "$MODEL" --ctx "$ctx" \
        --ncmoe "$ncmoe" --kv "$kv" \
        --n-predict "$N_PREDICT" --reps "$REPS" --warmup "$WARMUP" \
        --prompt-reps "$PROMPT_REPS" --out "$OUT/$name")
  if [ "$streams" -gt 1 ]; then
    args+=(--parallel "$streams" --concurrent "$streams")
  fi
  if [ "$op_offload" = "off" ]; then
    args+=(--extra --no-op-offload)
  fi

  "$PYTHON" -I "$ROOT/scripts/bench_serve.py" "${args[@]}" \
      2>&1 | tee "$OUT/$name.bench.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] || [ ! -f "$OUT/$name.metrics.json" ]; then
    echo "  run failed (rc=$rc)"
    failed=$((failed + 1))
    continue
  fi

  read -r dec pre agg vram < <("$PYTHON" -I -c '
import json, sys
m = json.load(open(sys.argv[1]))
timed = [r for r in m.get("runs", []) if not r.get("warmup")]
dec = [r["predicted_per_s"] for r in timed if r.get("predicted_per_s") is not None]
pre = [r["prompt_per_s"] for r in timed if r.get("prompt_per_s") is not None]
agg = [sum(x for x in (r.get("per_req_predicted_per_s") or []) if x is not None)
       for r in timed]
def avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")
vram = m.get("max_vram_mib") or 0
print(f"{avg(dec):.1f} {avg(pre):.1f} {avg(agg):.1f} {vram / 1024:.1f}")
' "$OUT/$name.metrics.json")
  if [ -z "$dec" ] || [ -z "$pre" ] || [ -z "$agg" ] || [ -z "$vram" ]; then
    echo "  could not parse $OUT/$name.metrics.json"
    failed=$((failed + 1))
    continue
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$ncmoe" "$streams" "$dec" "$pre" "$agg" "$vram" >> "$rows"
  ran=$((ran + 1))
  echo
done

if [ "$ran" -eq 0 ]; then
  echo "bench: no config completed" >&2
  exit 1
fi

echo "=== results ($OUT) ==="
"$PYTHON" -I - "$rows" <<'PY'
import sys

rows = [line.rstrip("\n").split("\t") for line in open(sys.argv[1]) if line.strip()]
out_rows = [(label, ncmoe, dec, pre, agg if int(streams) > 1 else "-", vram)
            for label, ncmoe, streams, dec, pre, agg, vram in rows]
header = ("workload", "ncmoe", "decode tok/s", "prefill tok/s", "aggregate tok/s", "VRAM GiB")
widths = [max(len(str(r[i])) for r in [header, *out_rows]) for i in range(len(header))]
fmt = "  ".join(f"{{:<{w}}}" for w in widths)
print(fmt.format(*header))
print(fmt.format(*("-" * w for w in widths)))
for row in out_rows:
    print(fmt.format(*row))
print()
print("decode/prefill are llama-server timings averaged over the timed runs;")
print("aggregate is the sum of concurrent per-request decode rates.")
PY

echo
if [ "$failed" -ne 0 ]; then
  echo "bench: $failed config(s) incomplete; see $OUT" >&2
  exit 1
fi
echo "bench: done"
