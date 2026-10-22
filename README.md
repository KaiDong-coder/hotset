# hotset

**A gpt-oss-120B inference engine for one 48 GB GPU and 30 GB of RAM.**

`gpt-oss-120B` is 117B parameters (5.1B active, 36 layers, 128 experts, top-4) in
MXFP4. Fitting it on a single RTX A6000 is not the hard part — llama.cpp can do
that. Serving it *fast* is, because at that size every token is a memory-
bandwidth problem: decode streams expert weights from host RAM over PCIe, and
whatever you do, the wall is bytes-per-token, not FLOPs.

hotset is built around that one fact. On a stock placement this box measured
**29 tok/s decode**; hotset measures **30.6 tok/s at `ncmoe=8`** — 92% of the
40.8 tok/s the host bandwidth of this machine allows. The remaining lever is to
read *fewer* bytes, and routing traces say how: a 14-expert-per-layer cache in
VRAM cuts host reads from 404.5 to 246.6 MiB/token, i.e. **50.3 tok/s**
(replayed from a real 897-token routing dump; the cache itself is in progress —
see [Status](#status)).

```bash
git clone https://github.com/<you>/hotset && cd hotset
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-cli llama-perplexity -j "$(nproc)"
python3 scripts/download_modelscope.py          # ~59 GiB, resumable
./hotset serve --ctx 8192                       # prints the config, then serves
```

`hotset serve` reads the GGUF tensor table and picks `-ngl`, `--n-cpu-moe`, KV
type and thread counts from **this machine's VRAM and `MemAvailable`** — it does
not assume the numbers below. The endpoint is
`http://127.0.0.1:8080/v1/chat/completions`.

## What is in the box

| piece | what it does |
|---|---|
| `hotset` | entry point: `serve`, `fit`, `bench`, `test`, `queue` |
| `scripts/moe_fit.py` | model-exact placement from the GGUF table; refuses configs that would thrash |
| `scripts/serve.sh` | `moe_fit --run`: print the chosen config, then exec `llama-server` |
| `scripts/bench.sh` | one command that reproduces the table below on your box |
| `scripts/routing_skew.py` | replay a routing dump: per-layer expert skew, LRU hit rate |
| `scripts/batch_union.py` | price expert caches and batching from a real trace (bytes/token → tok/s) |
| `scripts/gpu_queue.py` | wait for a VRAM budget on a **shared** GPU, run one command, never signal a foreign pid |
| `scripts/bench_queue.py` | the benchmarks that are waiting on VRAM, run sequentially |
| `scripts/membw.c` | the host streaming-read probe behind the 16.5 GB/s ceiling |
| `tests/run_tests.py` | placement unit tests + an A/B kernel-equivalence suite |
| `docs/PROBE-REPORT.md` | every measurement, A/B experiment and failure mode |
| `docs/PERF-LEVERS.md` | the byte model, calibrated against measured tok/s |

## Measured numbers

RTX A6000 48 GB, i9-10900X (10C/20T), 30 GB RAM, PCIe 3.0 x16. Server-side
`timings` from `llama-server`, greedy decode.

| workload | decode tok/s | prefill tok/s | placement |
|---|---|---|---|
| 8k ctx, f16 KV, 1 stream | **29–30.6** | **160** | `ncmoe=8`, `--no-op-offload` |
| 32k ctx, q8_0 KV, 1 stream | **24.9** | **179** | `ncmoe=8` |
| 8k ctx, f16 KV, 2 streams | **26.6 aggregate** | — | `ncmoe=8`, `--parallel 2` |

Stock `llama.cpp` at the same placement is slower (25.4 tok/s decode / 81 tok/s
prefill at `ncmoe=9`). Two things account for it: the MXFP4 `mul_mat_id` fast
paths in `ggml/src/ggml-cpu/ggml-cpu.c` (4-expert interleaved decode, multi-token
prefill), and the CPU-resident placement chosen by `moe_fit.py`.

The `30.6` is not a new peak, it is the *ceiling check*: decode reads
`ncmoe × 50.56 MiB` per token, this CPU streams 16.5 GB/s in one pass, and the
engine sustains 12.4 GB/s (92% of 40.8 tok/s). That is why the next section is
about bytes, not kernels.

## Where the speed is

`docs/PERF-LEVERS.md` derives every row below from `scripts/batch_union.py`
replaying a real routing dump; the `b=1` row reproduces the measured 30.6 tok/s
exactly, which is what makes the rest trustworthy.

| config | host bytes/token | decode tok/s | cost |
|---|---|---|---|
| `ncmoe=8`, no cache (today) | 404.5 MiB | 30.6 (measured) | — |
| `ncmoe=9` + 14 experts/layer cached | 246.6 MiB | **50.3** (projected) | 1.4 GiB VRAM |
| `ncmoe=10` + 28 experts/layer | — | **67.6** (projected) | 2.8 GiB VRAM + 3.2 GB host |

Batching is the other axis, and it is a *substitute*, not a complement: a
batched `mul_mat_id` reads each distinct expert once, so bytes scale with the
expert union — aggregate 2.67x at `b=32` — but with a 14-expert cache, `b=8`
(43.9 tok/s) is *slower* than `b=1` (50.3) because the wider union thrashes the
cache. Cache first; batch only at `b ≥ 16`.

## Quickstart

Prerequisites: NVIDIA driver + CUDA toolkit, `git`, `cmake ≥ 3.20`, `python3`;
~80 GB free disk (build tree + the 63.4 GB GGUF); one GPU with 48 GB VRAM and
**~19 GB of `MemAvailable`** for the CPU-resident experts. Measured on Ubuntu
26.04, CUDA 13.1.

```bash
git clone https://github.com/<you>/hotset && cd hotset

# 1. Build. llama.cpp's CMake tree is the repository root.
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-cli llama-bench -j "$(nproc)"

# 2. Model (~59 GiB, resumable). Needs `pip install --user requests`.
python3 scripts/download_modelscope.py

# 3. What this box should run — printed, not guessed:
./hotset fit --ctx 8192

# 4. Serve it (prints the placement, then starts llama-server):
./hotset serve --ctx 8192
```

```bash
curl -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "n_predict": 32, "temperature": 0}' \
  | python3 -m json.tool
```

### Reproduce the table

```bash
./hotset bench                 # every ctx/KV/stream pair, one row each
REPS=5 ./hotset bench          # more samples; results/bench-*.json
```

### Test

```bash
./hotset test                  # unit + kernel A/B (no model, no GPU needed)
./hotset test --e2e            # + perplexity and needle, needs the GGUF
CMAKE=/path/to/cmake ./hotset test   # if cmake is not on PATH
```

The kernel stage builds `test-mmid-cpu`, runs it with the fast paths on and
off, and compares elementwise. That comparison is deliberate: an in-kernel
self-check compares against a reference using the same pointers, which is
exactly how a `ry[s]` activation-indexing bug survived until the two paths were
diffed against each other.

### Sharing a GPU you do not own

Several measurements here were blocked by an unrelated tenant holding 41.9 of
49.1 GiB. hotset never kills or deprioritises someone else's process — it waits,
and leaves evidence when it cannot run:

```bash
python3 -I scripts/gpu_queue.py --need-mib 47700 -- <one command>
python3 -I scripts/bench_queue.py --list     # what is waiting on VRAM
```

## Status

- **v0.1, what is measured:** CPU-resident MoE placement, MXFP4 `mul_mat_id`
  fast paths (decode and prefill), the auto-placement planner, the routing-trace
  tooling, and the test suite.
- **In progress:** the split-MoE GPU expert cache (`--moe-cache N`). The design
  is in `docs/SPLIT-MOE.md` (masked-id skip in the CPU and CUDA kernels, two
  `mul_mat_id` ops instead of one). The 50.3 / 67.6 tok/s rows above are
  **byte-model projections from real traces**, not measurements — this README
  will not call them measured until they are.
- **Not available:** single-stream decode above 40.8 tok/s without reading fewer
  bytes (PCIe 3.0 at 16.5 GB/s is the wall).

## Honest limits

- **PCIe 3.0 is an assumption.** At Gen 3, computing MoE on the CPU is ~3.3x
  faster than streaming experts to the GPU (25.4 vs 7.7 tok/s at the same
  placement), which is why the planner passes `--no-op-offload`. `moe_fit.py`
  flips `op_offload` to `auto` above Gen 3, but that branch is unmeasured.
- **30 GB RAM is the first-order constraint, not VRAM.** `ncmoe=12` (19.4 GiB
  of CPU experts) evicted expert pages: 122 GB of disk reads for three requests
  and decode collapsed to 21.8 tok/s. The planner sizes CPU layers to
  `MemAvailable` and reports a shortfall instead of hiding it.
- **f16 KV is the default; q8_0 is a compromise.** q8_0 at 32k is the only way
  to fit that context with `ncmoe=8`, and it is slightly faster, but it failed
  the long-context needle test where f16 passed. Use it to fit, not to speed up.
- **One box's numbers.** The GPU is shared and back-to-back runs drift by ~7%;
  compare configs within a run, not across days. Your host bandwidth sets your
  ceiling — run `scripts/membw.c` before believing any tok/s figure.
- **Upstream tree.** This repository carries a llama.cpp checkout with local
  kernel changes; it is not a drop-in replacement for upstream. See
  `NOTICE.md`.

## Name

A *hot set* is the small, frequently-used subset of a large working set — here,
the ~14 of 128 experts per layer that absorb 54–68% of routed slots. hotset is
an engine built to find that set, keep it where it is cheap to read, and take
the rest from where it is not.

## License

MIT, as llama.cpp. See `LICENSE` (upstream) and `NOTICE.md` (what was changed).
