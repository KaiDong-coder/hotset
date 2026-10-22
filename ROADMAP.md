# Roadmap

**English** | [简体中文](ROADMAP.zh-CN.md)

Ordered by how much decode they buy per unit of work. Everything here is
measurable on one box; nothing here is scoped by ambition alone.

## v0.2 — the expert cache (the main line)

Today decode reads `ncmoe × 50.56 MiB` per token and is capped at 40.8 tok/s by
host bandwidth. The top 14 of 128 experts per layer absorb 54–68% of routed
slots, so an LRU of those experts in VRAM removes most of the reads:

| step | projected decode | state |
|---|---|---|
| masked-id skip in `ggml-cpu` and `ggml-cuda` | prerequisite | design only |
| `--moe-cache 14` at `ncmoe=9` | 50.3 tok/s | design only |
| `--moe-cache 28` at `ncmoe=10` | 67.6 tok/s | needs 3.2 GB more host RAM |

The design, with file:line targets, is [`docs/SPLIT-MOE.md`](docs/SPLIT-MOE.md).
The acceptance test that matters: splitting one matmul into a CPU half and a GPU
half, both masked and zero-filled, must reproduce the unmasked op to max rel err
≤ 1e-5 — **on CPU alone**, so it runs in CI without a GPU.

**How to verify:** `./hotset bench` at `--moe-cache 0/14/28`; the margin is real
only if server `timings` beat the 30.6 tok/s baseline by more than the ~7%
run-to-run drift.

## Next — continuous batching, but only at b ≥ 16

A batched `mul_mat_id` reads each distinct expert once, so host bytes scale with
the union: 2.67x aggregate at `b=32`. But with a 14-expert cache, `b=8` is
*slower* than `b=1` (43.9 vs 50.3) because the wider union thrashes the cache.
Land the cache first; then batch at b ≥ 16, where the union is still smaller than
the cache's working set.

## Hardware coverage — the cheapest way to contribute

Every number in this repo comes from **one** machine: RTX A6000 48 GB, i9-10900X,
DDR4 in what looks like dual-channel (16.5 GB/s measured, ~19% of the platform's
theoretical 93.8 GB/s), PCIe 3.0 x16.

What would change the policy, and what we want measured:

- **A quad-channel or 8-channel host.** If your host streams 30 GB/s, the
  ceiling at `ncmoe=8` moves from 40.8 to ~74 tok/s and the cache becomes less
  urgent. `scripts/membw.c` is the probe.
- **PCIe 4.0/5.0.** `moe_fit.py` switches `op_offload` to `auto` above Gen 3.
  That branch is untested; streaming experts to the GPU may win at Gen 4.
- **A 24 GB card (3090/4090).** The placement will be a different shape
  (`ncmoe` ≥ 14) and host RAM becomes the binding constraint sooner. `moe_fit`
  should report a shortfall, not a crash — confirm that.
- **AMD (ROCm) or Metal.** The MXFP4 fast paths are CPU-side and portable; the
  CUDA-specific masked-id work is not. Untested on both.

Open an issue titled `hardware: <gpu> + <cpu> + <ram>` with `./hotset fit --json`
and `scripts/membw.c` output. Reports that contradict this README are welcome;
they will be folded in, with the machine named.

## Later

- **Split KV precision** (K at higher precision than V): q8_0 KV is the only way
  to fit 32k here, and it lost the needle test where f16 passed. Per-head scales
  should buy the same 2.4 GiB without the retrieval loss
  ([`docs/PERF-LEVERS.md`](docs/PERF-LEVERS.md), lever 3).
- **Overlap CPU MoE with GPU work.** Currently ~26 of 34.5 ms/token is CPU MoE
  serialized before ~8 ms of GPU work. The cache shrinks the CPU side, which
  makes overlap worth more, not less.

## Not doing

- Distributed / multi-node serving. One box is the point.
- A serving framework, auth, or billing. Use your reverse proxy.
- Upstream PRs from this fork. Local kernel changes stay local
  ([`NOTICE.md`](NOTICE.md)).
