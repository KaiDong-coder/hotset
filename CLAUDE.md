# hotset — notes for agents working in this repository

This is a **fork of ggml-org/llama.cpp** serving gpt-oss-120B with CPU-resident
MoE. Upstream's contribution rules in `AGENTS.md` govern patches sent to
ggml-org/llama.cpp; they do not govern changes kept in this fork.

## Layout

- `scripts/` — placement (`moe_fit.py`, `serve.sh`), benchmarks (`bench.sh`,
  `bench_serve.py`), routing analysis (`routing_skew.py`, `batch_union.py`),
  shared-GPU queueing (`gpu_queue.py`, `bench_queue.py`). Stdlib only.
- `tests/run_tests.py` — unit stage (discovers `tests/test_*.py`) + kernel A/B
  stage (`test-mmid-cpu`, fast paths on vs off). Run from the repo root.
- `docs/PROBE-REPORT.md` — the evidence trail. If a number here disagrees with
  the report, the report is right.
- Upstream llama.cpp code is untouched except the MXFP4 `mul_mat_id` paths in
  `ggml/src/ggml-cpu/ggml-cpu.c` (see `NOTICE.md`).

## Invariants

- Never kill, signal, or renice a process hotset did not start. On a shared GPU,
  wait with `scripts/gpu_queue.py` instead.
- Default-off features stay default-off: flag off must produce the upstream
  graph byte-for-byte.
- A projection is labelled a projection. Do not promote a replayed-trace number
  to "measured" until it comes from a server `timings` block.
- Run `python3 -I` for scripts that read model files or traces.
