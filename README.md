# hotset

[![tests](https://github.com/KaiDong-coder/hotset/actions/workflows/tests.yml/badge.svg)](https://github.com/KaiDong-coder/hotset/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**English** | [简体中文](README.zh-CN.md)

**gpt-oss-120B (117B parameters, 5.1B active) served on one 48 GB GPU and 30 GB
of RAM — 30.6 tok/s decode, 97.8 tok/s prefill, and a measured byte model that
tells you what the next 20 tok/s costs.**

A 117B model on one RTX A6000 is a memory-bandwidth problem, not a FLOP problem:
every token streams expert weights from host RAM over PCIe. hotset sizes the
split to the machine, runs the MoE on tuned MXFP4 kernels, and ships the trace
replay tools that price the remaining levers instead of guessing at them.

## This box, measured

| | | what sets it |
|---|---|---|
| decode, 8k ctx, f16 KV | **30.6 tok/s** | 92% of the 40.8 tok/s this host's memory allows |
| prefill, 8k ctx | **97.8 tok/s** | CPU MoE (`--no-op-offload`), 9.1 cores busy |
| VRAM | 47.7 of 49.1 GiB | 8 of 36 MoE layers on the CPU |
| host RAM | 17.3 of 30 GB | `MemAvailable`, not VRAM, is the first constraint |

RTX A6000 48 GB, i9-10900X (10C/20T), PCIe 3.0 x16. Every row is
`llama-server`'s own `timings` block, greedy decode; `./hotset bench`
reproduces it.

| other workloads | decode tok/s | prefill tok/s |
|---|---|---|
| 32k ctx, q8_0 KV, 1 stream | 24.9 | 179 |
| 8k ctx, f16 KV, 2 concurrent streams | 26.6 aggregate | — |

Placement matters as much as kernels: the *same* box at `ncmoe=9` gives 25.4
tok/s decode / 81.1 tok/s prefill — one more layer on the GPU is worth 20%. The
kernel fast paths are worth another 11.6% at equal placement (1.093 vs 1.236 ms
per `mul_mat_id`, measured).

## The wall, and the way past it

```
  decode tok/s
   70 ┤                                      ┌─ 67.6   ncmoe=10 + 28-expert cache   (projected)
   60 ┤                               ┌──────┘
   50 ┤                        ┌──────┘ 50.3          ncmoe=9 + 14-expert cache    (projected)
   40 ┤                 ┌──────┘ 40.8                ceiling: 16.5 GB/s host bandwidth
   30 ┤───────●─────────┘ 30.6                       measured today, ncmoe=8
   20 ┤
      └──────────────────────────────────────────────────────────────────▶
         404.5 MiB/token        246.6                      fewer bytes, not more FLOPs
```

Decode reads `ncmoe × 50.56 MiB` per token. At this host's measured 16.5 GB/s
that caps decode at 40.8 tok/s, and the engine already sustains 12.4 GB/s — 92%
of it. No further kernel work moves that number; only reading fewer bytes does.
Routing is skewed enough to allow it: per layer, the top 14 of 128 experts
absorb 54–68% of routed slots (uniform would be 10.9%).

`scripts/batch_union.py` replays a real 897-token routing dump to price that
cache; its `b=1` row reproduces the measured 30.6 tok/s exactly, which is what
makes the projected rows worth publishing. `docs/PERF-LEVERS.md` has the full
derivation, including why batching is a *substitute* for the cache rather than a
complement.

## What is in the box

| piece | what it does |
|---|---|
| `hotset` | entry point: `serve`, `fit`, `bench`, `test`, `queue` |
| `scripts/moe_fit.py` | model-exact placement from the GGUF table; refuses configs that would thrash |
| `scripts/serve.sh` | `moe_fit --run`: print the chosen config, then exec `llama-server` |
| `scripts/bench.sh` | one command that reproduces the tables above on your box |
| `scripts/routing_skew.py` | replay a routing dump: per-layer expert skew, LRU hit rate |
| `scripts/batch_union.py` | price expert caches and batching from a real trace (bytes/token → tok/s) |
| `scripts/gpu_queue.py` | wait for a VRAM budget on a **shared** GPU, run one command, never signal a foreign pid |
| `scripts/bench_queue.py` | the benchmarks that are waiting on VRAM, run sequentially |
| `scripts/membw.c` | the host streaming-read probe behind the 16.5 GB/s ceiling |
| `tests/run_tests.py` | placement unit tests + an A/B kernel-equivalence suite |
| [`docs/PROBE-REPORT.md`](docs/PROBE-REPORT.md) | every measurement, A/B experiment and failure mode |
| `docs/PERF-LEVERS.md` | the byte model, calibrated against measured tok/s |
| `docs/SPLIT-MOE.md` | the design for the expert cache (v0.2, in progress) |

## Quickstart

Prerequisites: NVIDIA driver + CUDA toolkit, `git`, `cmake ≥ 3.20`, `python3`;
~80 GB free disk (build tree + the 63.4 GB GGUF); one GPU with 48 GB VRAM and
**~19 GB of `MemAvailable`** for the CPU-resident experts. Measured on Ubuntu
26.04, CUDA 13.1.

```bash
git clone https://github.com/KaiDong-coder/hotset && cd hotset

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

## Is this for you?

- **Yes**, if you have one 48 GB GPU and want to serve a 120B-class model from a
  box you already own, and you want to know where its speed actually comes from.
- **Yes**, if you work on MoE inference and want a calibrated byte model and a
  trace-replay harness you can point at your own hardware — that is most of what
  is in this repository.
- **Not yet**, if you need many concurrent users: aggregate throughput at 2
  streams is 26.6 tok/s, only ~8% above one stream. Continuous batching is a
  [`docs/PERF-LEVERS.md`](docs/PERF-LEVERS.md) projection, not a shipped feature.

## Status

- **v0.1, measured:** CPU-resident MoE placement, MXFP4 `mul_mat_id` fast paths
  (decode and prefill), the auto-placement planner, the routing-trace tooling,
  and the test suite.
- **In progress:** the split-MoE GPU expert cache (`--moe-cache N`). The design
  is in [`docs/SPLIT-MOE.md`](docs/SPLIT-MOE.md) (masked-id skip in the CPU and
  CUDA kernels, two `mul_mat_id` ops instead of one). The 50.3 / 67.6 tok/s rows
  above are **byte-model projections from real traces**, not measurements — this
  README will not call them measured until they are.
- **Not available:** single-stream decode above 40.8 tok/s without reading fewer
  bytes (PCIe 3.0 at 16.5 GB/s is the wall).

See [`ROADMAP.md`](ROADMAP.md) for what v0.2 needs and where contributions help
most.

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
  [`NOTICE.md`](NOTICE.md).

## Contributing

Measurements on hardware this repository has not seen are the most useful thing
you can bring — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[issue #2](https://github.com/KaiDong-coder/hotset/issues/2). Questions and
half-formed results belong in
[Discussions](https://github.com/KaiDong-coder/hotset/discussions).

## Name

A *hot set* is the small, frequently-used subset of a large working set — here,
the ~14 of 128 experts per layer that absorb 54–68% of routed slots. hotset is
an engine built to find that set, keep it where it is cheap to read, and take
the rest from where it is not.

## License

MIT, as llama.cpp. See [`LICENSE`](LICENSE) (upstream) and
[`NOTICE.md`](NOTICE.md) (what was changed).
