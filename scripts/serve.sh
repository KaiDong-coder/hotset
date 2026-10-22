#!/usr/bin/env bash
# serve.sh — auto-fit the placement for this machine, print it, start the server.
#
# Thin wrapper around `moe_fit.py --run`: moe_fit reads the GGUF tensor table,
# chooses -ngl / --n-cpu-moe / KV / threads from VRAM and MemAvailable, prints
# the chosen config and then execs llama-server (retrying one CPU layer further
# out if VRAM allocation fails).
#
# Usage:
#   ./scripts/serve.sh [moe_fit options...]
#
# Common overrides (environment, so they do not collide with passthrough args):
#   MODEL=path/to/model.gguf      default models/gpt-oss-120b-MXFP4.gguf
#   LLAMA_SERVER=path/to/bin      default build/bin/llama-server
#   CTX=8192                      context size
#   KV=f16                        KV cache dtype: f16 (default) or q8_0
#
# Any extra arguments are passed to moe_fit.py (e.g. --ctx 32768 --kv-dtype q8_0).
# llama-server's default bind address is http://127.0.0.1:8080.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$ROOT/models/gpt-oss-120b-MXFP4.gguf}"
BIN="${LLAMA_SERVER:-$ROOT/build/bin/llama-server}"
CTX="${CTX:-8192}"
KV="${KV:-f16}"

if [ ! -f "$MODEL" ]; then
  echo "serve: model not found: $MODEL" >&2
  echo "serve: run scripts/download_modelscope.py or set MODEL=..." >&2
  exit 1
fi
if [ ! -x "$BIN" ]; then
  echo "serve: llama-server not found: $BIN" >&2
  echo "serve: build it (see README) or set LLAMA_SERVER=..." >&2
  exit 1
fi

echo "serve: model  $MODEL"
echo "serve: server $BIN"
echo "serve: choosing placement for ctx=$CTX kv=$KV (this reads the GGUF header)"
echo "serve: endpoint will be http://127.0.0.1:8080/completion"
echo

# `-I` keeps the interpreter isolated; "--run" prints the config and starts the
# server.  Extra args come last so they can override the env-derived defaults.
exec python3 -I "$ROOT/scripts/moe_fit.py" \
  --model "$MODEL" \
  --bin "$BIN" \
  --ctx "$CTX" \
  --kv-dtype "$KV" \
  --run "$@"
