# Where the speed is, measured and projected

Date: 2026-10-22
Hardware: RTX A6000 48 GB, i9-10900X (10C/20T), 30 GB RAM, PCIe 3.0 x16.
Measured single-pass host read: 16.5 GB/s; the engine achieves 12.4 GB/s at
ncmoe=8 (30.6 tok/s x 404.5 MiB/token).

Every projection below comes from replaying real routing traces with
`scripts/batch_union.py`; the b=1 row of that replay reproduces the measured
30.6 tok/s exactly, which is what makes the other rows trustworthy.

## Re-measuring on a shared card

The card is shared with an unrelated tenant (`/opt/strata/engine/strata`,
41.9 of 49.1 GiB for most of 2026-10-21/22). Never stop it, and do not start a
benchmark on a single free reading — queue the measurements instead:

```bash
python3 -I scripts/bench_queue.py --list   # what is waiting on VRAM
python3 -I scripts/bench_queue.py          # wait for the card, then run all
python3 -I scripts/gpu_queue.py --need-mib 47700 -- <one command>
```

`scripts/bench_queue.py` declares the waiting benchmarks (PROBE-023's
fast/generic/warm table; the split-MoE A/B at cache 0/14/28 once PROBE-025
lands) and hands each one to `scripts/gpu_queue.py`, sequentially — never two
GPU jobs at once. `gpu_queue.py` requires the budget (47.7 GiB for the
measured ncmoe=8 placement) to be free for 3 consecutive samples 30 s apart,
then runs exactly one command. If it never frees, it writes
`results/gpu-queue/<name>.json` naming the resident pid and its MiB, so "we
could not measure" is evidence rather than an empty log.

The projections below can therefore be re-measured on the next free window
with one command; the split-MoE rows stay blocked until PROBE-025 lands the
`--moe-cache N` flag.

## The ceiling

Decode reads `ncmoe x 50.56 MiB` per token. At ncmoe=8 that is 404.5 MiB, so
the host bandwidth alone caps decode at **40.8 tok/s**. The kernel already runs
at 92% of that (finding 8), so no further kernel work moves the needle. The
only way faster is **fewer bytes**.

## Lever 1: GPU-resident expert cache (single-stream)

Routing is skewed: per layer the top-14 experts absorb 54-68% of the routed
slots (uniform would be 10.9%). An LRU of 14 experts per layer is therefore a
hit-rate machine, and a hit costs zero host bytes.

| config | host bytes/token | decode tok/s | VRAM cost |
|---|---|---|---|
| ncmoe=8, no cache (today) | 404.5 MiB | 30.6 (measured) | - |
| ncmoe=9 + 14 experts/layer | 246.6 MiB | **50.3** | 1.4 GiB |
| ncmoe=10 + 28 experts/layer | - | **67.6** | 2.8 GiB + 3.2 GB host |

The cache has to be paid for out of VRAM, so it trades one resident layer for
the cache (1.4 GiB is about one layer's 1617.9 MiB). Break-even is an 11.1%
hit rate; the measured skew is 4-5x above it. The design is in
`docs/SPLIT-MOE.md` (PROBE-014).

## Lever 2: continuous batching (aggregate throughput)

A batched `mul_mat_id` reads each distinct expert's block once per batch, so
host bytes scale with the *union*, not with `tokens x top_k`:

| batch | union/layer | bytes/token | aggregate tok/s | vs b=1 |
|---|---|---|---|---|
| 1 | 4.00 | 404.5 MiB | 30.7 | 1.00x |
| 2 | 7.10 | 359.1 MiB | 34.5 | 1.13x |
| 4 | 12.20 | 308.3 MiB | 40.2 | 1.31x |
| 8 | 20.33 | 257.0 MiB | 48.2 | 1.57x |
| 16 | 32.15 | 203.2 MiB | 61.0 | 1.99x |
| 32 | 48.01 | 151.7 MiB | 81.7 | **2.67x** |

This is **aggregate** throughput for concurrent requests, not per-stream
latency: at b=8 a batch takes ~165 ms for 8 tokens, so each stream sees only
~6 tok/s while the box serves 48 tok/s. Right for a shared endpoint, wrong for
one chat session.

### The two levers are substitutes, not complements

With a 14-expert cache, batch 8 is *slower* than batch 1 (43.9 vs 50.3
tok/s): a larger batch widens the expert union and thrashes the small cache.
The cache already captures the skew that batching would exploit. Order of
work: cache first, then batch only at b >= 16.

### The cache raises the value of overlap

Today 26.2 of 34.5 ms/token is CPU MoE and ~8 ms is GPU work serialized after
it. After the cache, CPU MoE drops to ~16 ms and the GPU share grows, so
overlapping the two (graph split / concurrent requests) becomes the next
lever instead of a rounding error.

## Lever 3: KV quantisation (memory and a little speed)

At 32k ctx, f16 KV costs ~4.8 GiB. The only way to fit 32k on this card today
is q8_0 (2.4 GiB), which we measured losing needle retrieval (finding 10).
A KV scheme that keeps K at higher precision than V (per-head scales) should
buy the same 2.4 GiB without the retrieval loss, and those 2.4 GiB are worth
~1.5 layers on the GPU, i.e. another ~14% decode on their own.

## What is not available

- Single-stream decode cannot exceed 40.8 tok/s without reducing bytes; PCIe
  3.0 at 16.5 GB/s is the wall.
- Host RAM is the second wall: ncmoe=10 needs 16.2 GiB and ncmoe=12 (19.4 GiB)
  thrashed with 122 GB of disk reads. Any cache plan that moves many more
  layers to the CPU fails here.
