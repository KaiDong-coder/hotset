# Split MoE matmul: cached experts on the GPU, the rest on the CPU

Date: 2026-10-22. Task: PROBE-014.

**Status: design + measurement. Decision: PROCEED**, with one correction to the
obvious implementation: the cache must be filled once per request (calibrated
on the prompt / first decode tokens), not by inserting every miss. The
insert-on-miss policy uploads the same number of bytes it saves per token, and
on PCIe 3.0 that upload alone costs more than the win.

## TL;DR

- Measured hit rate for a **14-experts-per-layer** cache on the 9 CPU-expert
  layers at ncmoe=9: **44.4%** (calibrated static set, 300-token blocks,
  layers 0-7, mean of the route/code/math prompts; 37-63% per prompt). The
  dynamic LRU estimate is 45.9%; layer 8, measured separately, is ~35%.
- **Per-token bytes: 253 MiB/token** read from host RAM by the CPU for the
  uncached experts, down from 455 MiB/token with no cache. The GPU supplies
  the other 202 MiB/token from VRAM. A one-time fill of the cache costs
  1.56 GiB (~220 ms at 7.6 GB/s).
- Projected decode at ncmoe=9: 16.3 ms CPU + 9.2 ms GPU = **25.5 ms/token,
  ~39 tok/s** vs 29 tok/s at ncmoe=8 (+35%). With CPU/GPU overlap the ceiling
  is 45-50 tok/s, but that needs a second stream (not part of this task).
- The current cache policy inserts (and uploads) every miss. At 44.4% that is
  the same 253 MiB/token crossing PCIe, ~35 ms/token at the measured 7.6 GB/s,
  i.e. the whole win is lost. A static cross-prompt hot set is not an
  alternative either: a top-14 set calibrated on two topics hits only ~10% on
  the third, because the hot set is prompt-specific (~98-100 distinct experts
  per layer over 300 tokens). The viable policy is a **per-request
  calibration**: observe the routing for the first ~20-50 tokens (or the
  prompt prefill), upload the top-k per layer once, then keep the set fixed.
  Measured hit rate of that policy: 44.4% at k=14 (W=50), 41.5% at k=12.
- Everything below the 34.5 ms/token (29 tok/s) abandon bar is the same as
  v0.1; the projected 25.5-26.4 ms/token clears it.

## 1. Where the graph splits today

Per token each MoE layer is three `mul_mat_id` ops (gate, up, down). The
cache already forces a scheduler split at exactly the right place:

1. `build_moe_ffn` (`src/llama-graph.cpp:2152`) produces
   `selected_experts` `[4, T]`, then calls `build_moe_cache_slots`
   (`llama-graph.cpp:2319`).
2. `build_moe_cache_slots` (`llama-graph.cpp:2532`) asks
   `llama_moe_cache::get_slot_map(il, n_tokens, n_expert_used)`
   (`llama-moe-cache.cpp:496`). For decode (`n_tokens <= 32`) it returns the
   layer's host-resident slot map, and the function emits
   `ggml_get_rows(slot_map, ids)` (`llama-graph.cpp:2565`). Because the slot
   map is a host weight, that node starts a new scheduler split.
3. `build_lora_mm_id` (`llama-graph.cpp:1669`) emits `mul_mat_id` against
   `moe_cache->get_experts(w)` (the GPU bank) with the resulting slots
   (`llama-graph.cpp:1677-1679`). The same slots tensor feeds gate, up and
   down, so the split is once per layer per token.
4. At execution the copy callback `sched_copy_experts`
   (`llama-context.cpp:2762`) runs `llama_moe_cache::copy`
   (`llama-moe-cache.cpp:654`): it reads `selected_experts` D2H, plans the
   SLRU (`plan`, `llama-moe-cache.cpp:108`), uploads every miss
   (`llama-moe-cache.cpp:600-615`), and rewrites the slot map so each routed
   expert points at its bank slot.

So today the cache only avoids *re-uploading* experts; every routed expert is
still computed on the GPU from the bank, and every miss is uploaded and
computed on the GPU. The CPU MoE kernels in the graph are used only when the
cache is off.

One structural detail matters for the design: the cache is a **single shared
SLRU pool per (GPU, tensor-layout) group**, not a per-layer cache. With nine
CPU-expert layers and `--moe-cache-mib 1400`, the pool is ~110 slots shared by
all nine layers (~12 per layer on average; the keys are `(layer, expert)`,
`llama-moe-cache.cpp:21,37-48`). The per-layer "14 experts" number used in the
measurements below is the budget-equivalent reading; a shared LRU of 112
slots on the 8-layer trace hits 45.7% vs 46.0% for eight independent LRU(14)
caches, so the distinction does not change the conclusion.

## 2. Proposed graph

For every layer whose experts are in host memory (`il < ncmoe`), decode only:

```
# after the cache has planned this token (existing copy callback)
ids_cpu  = selected_experts;  ids_cpu [slot is a hit ] = -1   # misses keep the id
slot_gpu = planned_slots;     slot_gpu[slot is a miss] = -1   # hits keep the bank slot

res_cpu = mul_mat_id(host_w, cur, ids_cpu)    # uncached experts, CPU
res_gpu = mul_mat_id(bank,   cur, slot_gpu)   # cached experts, GPU
res     = res_cpu + res_gpu                   # both zero-filled before the op
```

The same hit/miss masks are reused for gate, up and down, because the routing
is one `selected_experts` tensor per layer. Both partial results must be
zero-filled: each kernel writes only rows belonging to the experts it
processes, and the `add` must not see stale rows for the skipped `-1` slots.

**CPU/GPU split ratio** at ncmoe=9, 14 experts/layer (h = 44.4%):

| quantity | CPU (misses) | GPU (hits) |
|---|---|---|
| routed expert slots per token (9 layers x 4) | 20.0 | 16.0 |
| expert bytes per token | 253.0 MiB from host RAM | 202.0 MiB from VRAM |
| MoE time per token | 16.3 ms | ~0.9 ms added to the ~8.3 ms GPU tail |
| PCIe | 0 (after the one-time fill) | - |

At the deployable size (12/layer, h = 41.5%): 21 CPU slots / 15 GPU slots,
266.2 MiB host / 188.8 MiB VRAM.

## 3. What has to change

1. **Negative-id skip in the CPU kernel.** `ggml_compute_forward_mul_mat_id`
   groups rows with `assert(i02 >= 0 && i02 < n_as)`
   (`ggml/src/ggml-cpu/ggml-cpu.c:1832`). A `-1` id must mean "no rows for
   this token", not an abort. The same test must exclude negative ids from
   `matrix_row_counts` and from the IL4 decode fast path (`ggml-cpu.c:1854`,
   which currently requires all four slots to be valid).
2. **Negative-id skip in ggml-cuda.** `mm_ids_helper`
   (`ggml/src/ggml-cuda/mmid.cu:33`) computes `nex_prev += expert_used <
   expert`; a negative id sorts below every expert and corrupts the compacted
   expert bounds. The fallback `ggml_cuda_mul_mat_id`
   (`ggml/src/ggml-cuda/ggml-cuda.cu:2042`) asserts
   `expert_to_use >= 0 && expert_to_use < ne02` at line 2118. Both must skip
   `-1` (and leave those rows unwritten).
3. **Zero-fill both destinations** before the op: `memset` on the CPU and
   `cudaMemsetAsync` on the CUDA stream, or an equivalent pre-zeroed tensor.
4. **Masks per token.** `llama_moe_cache` already computes hits and fills once
   per layer per token (`plan`, `llama-moe-cache.cpp:108`). Extend the copy
   callback to write two host tensors instead of only the slot map: the
   miss-ids for the CPU op and the hit-slots for the GPU op. The graph nodes
   consume them in the same split the current slot map already forces.
5. **Dependency, and why this is add, not overlap.** `cur` is a GPU tensor
   (the non-expert ops stay offloaded with `-ngl 99`), so the CPU op needs a
   D2H copy of `cur` (2880 floats, 5.6 KiB per layer per token). In the
   current single-stream scheduler that copy plus the `add` serializes CPU
   and GPU work: the gain is additive (25.5 ms), not max(CPU, GPU). Reaching
   the 45-50 tok/s estimate needs a second CUDA stream and an event so the
   GPU half of layer l runs while the CPU half computes, plus the CPU MoE of
   layer l running under the GPU's layer l+1 attention. That is a scheduler
   change and is out of scope here.

Prefill is left on the existing CPU path (`n_tokens > 32` already disables
the slot map, `llama-moe-cache.cpp:496-510`); the split is a decode feature.

## 4. Measurements

### Method, and why it is not a plain ncmoe=9 run

The box has one 48 GB GPU, and during this task ~41.9 GiB of it were held by
an unrelated production service (`/opt/strata/engine/strata`) that must not be
touched. A real ncmoe=9 run needs 46.6 GiB, so the layer-8 routing was
measured with an all-CPU-expert run instead:

```
python3 scripts/quality_gate.py --bin build/bin/llama-server \
  --model models/gpt-oss-120b-MXFP4.gguf --ctx 2048 --ncmoe 0 --prompts route \
  --n-predict 40 --env GGML_MMID_DUMP=5000 --extra --cpu-moe -ub 64 -b 64
```

With `-ngl 99 --cpu-moe`, every layer's non-expert tensors (attention, norms,
router) are on the GPU exactly as at ncmoe=9, and the expert matmuls of
layers 0-8 are on the CPU exactly as at ncmoe=9. The compute placement for
layers 0-8 is therefore identical to ncmoe=9; only layers 9-35 differ (their
experts run on the CPU instead of the GPU). The trace has 39 decode tokens
(1404 layer-ops = 39 x 36).

Alignment against the existing ncmoe=8 trace (`results/quality/route_dump3.log`,
same route prompt) was verified by matching layer 0-7 ids: after dropping the
3 triples emitted around server start-up / model load at the top of the log,
the first 12 tokens of the request match with 22-28 of 32 slot ids per token
identical. The differences are slot-order permutations and one near-tie flip
per layer, consistent with a slightly different float reduction order (the
two runs used different `-ub`/`-b` settings), not with a misalignment. (The
three matrices of a layer always emit the same ids, so a "triple" of dump
lines is one layer-op.)

The 300-token steady-state numbers come from `route_dump3.log` at ncmoe=8:
three prompts x 300 tokens, 21,528 MMIDEXP lines = 7,176 layer-ops = 897
complete tokens. At ncmoe=8 its CPU layers are exactly layers 0-7, i.e. the
same layers that are on the CPU at ncmoe=9.

The cache model is a plain LRU (the statistic PROBE-013 used) and, for the
recommended policy, a static top-k set calibrated on the first W tokens of
the same continuation. The shipped cache is an SLRU; a protected segment
should be at least as good as LRU, but that has to be confirmed in situ (the
cache logs hits/misses/uploaded bytes at teardown).

### Hit rate, 14 experts/layer

300-token blocks, layers 0-7 (ncmoe=8 trace):

| prompt | tokens | top-14 coverage | LRU(14) hit | static top-14, calibrated on first 50 tokens |
|---|---|---|---|---|
| route | 300 | 53.6% | 46.1% | 37.1% |
| code | 300 | 67.7% | 54.4% | 62.9% |
| math | 297 | 46.6% | 36.8% | 33.2% |
| mean | - | - | 45.9% | **44.4%** |

Layer 8 (the ninth CPU layer at ncmoe=9) is not in the ncmoe=8 trace. On the
39-token all-CPU trace the same static calibration (W=10) gives 35.3% for
layer 8 and 38.1% for layers 0-8; the dynamic LRU(14) gives 32.7%. Taking
44.4% for layers 0-7 and 35% for layer 8 gives **~43% for the 9-layer
ncmoe=9 cache**; the doc uses 44.4% for the headline projection and notes the
9-layer number is 1-2 points lower.

A static set calibrated on *other prompts* is not viable. Evaluating a top-14
set trained on the other two topics on the third gives 10.9% (route), 9.6%
(code) and 9.0% (math); the per-layer working set is 78-117 distinct experts
over 300 tokens. The skew is *within* a prompt, not global, which is why the
calibration has to be per request.

### Per-token bytes

At ncmoe=9 with a 14/layer cache and h=44.4%:

| stream | bytes/token |
|---|---|
| CPU reads the 20 uncached expert slots from host RAM | **253.0 MiB** |
| GPU reads the 16 cached expert slots from VRAM | 202.0 MiB |
| host -> GPU upload (once per calibration, not per token) | 1.56 GiB one-time |
| without the cache (baseline ncmoe=9) | 455.0 MiB from host RAM |

At the deployable 12/layer (1.33 GiB, h=41.5%): 266.2 MiB/token host reads.
For the 9-layer estimate (~43% including layer 8) it is 259.4 MiB/token.
The byte accounting is per expert slot; an expert is 1617.9/128 = 12.64 MiB
(gate+up+down), i.e. 4.21 MiB per matrix, and 9 layers x 4 slots x 3 matrices
x 4.21 MiB = 455.0 MiB/token when nothing is cached.

**The upload policy is the deciding factor.** Inserting every miss (what
`llama_moe_cache::copy` does today) uploads the same 253 MiB/token that the
split saves, over PCIe 3.0. At the measured 7.6 GB/s that is ~35 ms/token:
serialized it gives ~60 ms/token, and even fully overlapped the token is
PCIe-bound at ~35 ms, i.e. no better than the 29 tok/s baseline. The cache
therefore has to be filled off the critical path: calibrate on the prompt
prefill or the first 20-50 decode tokens, upload the top-k per layer once
(~220 ms for 1.56 GiB at 7.6 GB/s, amortized over the response), and do not
insert during the steady state. Calibrating on the first 20 decode tokens
costs those tokens the baseline rate plus the fill: a 300-token response
averages (20*39.4 + 220 + 280*25.5)/300 = 27.2 ms/token, still 37 tok/s.

## 5. Timing projection

Baseline: ncmoe=8 = 34.5 ms/token (29 tok/s); ncmoe=9 = 39.4 ms/token
(25.4 tok/s measured). The CPU part is 27 ops x 1.09 ms = 29.4 ms at
ncmoe=9, the GPU tail ~8.3 ms (PROBE-REPORT findings 6, 8, 11).

| policy | h | CPU MoE | GPU | total | tok/s | vs 29 tok/s |
|---|---|---|---|---|---|---|
| ncmoe=8 baseline | - | 26.2 ms | 8.3 ms | 34.5 ms | 29.0 | - |
| ncmoe=9, no cache | 0 | 29.4 ms | 8.3 ms | 37.7 ms | 26.5 | -9% |
| 14/layer static, ncmoe=9 | 44.4% | 16.3 ms | 9.2 ms | **25.5 ms** | **39.2** | +35% |
| 12/layer static, ncmoe=9 | 41.5% | 17.2 ms | 9.2 ms | **26.4 ms** | **37.9** | +31% |
| with CPU/GPU overlap (needs a second stream) | 44.4% | max(16.3, 9.2) | - | 20-22 ms | 45-50 | +55-72% |

The break-even for trading the ninth GPU layer for the cache is h=11.1%
(finding 11). The measured 44.4% is 4.0x that, and even the layer-8-only
35.3% is 3.2x. The 34.5 ms/token abandon bar is cleared by 26%.

## 6. Decision

**Proceed to implementation**, with the split plus a per-request calibrated
static fill. Do not ship the current insert-on-every-miss policy: its measured
upload volume (253 MiB/token) costs ~33 ms/token on PCIe 3.0 and cancels the
gain. Keep the 34.5 ms/token end-to-end gate as the abandon criterion for the
implementation.

Concretely, the implementation should be attempted in this order:

1. Extend `llama_moe_cache` with two modes: `plan` a *static* top-k per layer
   from a calibration window, and return the hit/miss masks. The current SLRU
   stays as a fallback for comparison.
2. CPU negative-id skip + zero-fill, with a `test-mmid-cpu` case that compares
   the masked split sum against the generic unmasked op (max rel err expected
   ~1e-7; the existing harness at `tests/test-mmid-cpu.cpp` and
   `scripts/compare_mmid.py` is the A/B reference).
3. CUDA negative-id skip + zero-fill (the fallback and `mm_ids_helper`).
4. Graph: two `mul_mat_id` + `add` per matrix for `n_tokens == 1` and
   cache-layer experts; prefill unchanged.
5. Measure end to end at ncmoe=9 with 12/layer (1.33 GiB; the 14/layer
   1.56 GiB does not fit: ncmoe=9 peaks at 46.6 GiB of 48.0 GiB) and with
   `--no-op-offload`, both with and without a second stream.

## 7. Risks and open questions

- **VRAM fit.** 14/layer = 126 slots x 12.64 MiB + padding = 1.56 GiB;
  ncmoe=9 measured 46.6 GiB peak, leaving ~1.4 GiB. 12/layer allocates
  1.35 GiB (108+1 slots) — about 50-70 MiB of headroom, very tight.
  11/layer (1.23 GiB) is the safe start unless ctx or KV is reduced. The
  shared pool means the budget is expressed as `--moe-cache-mib`, not
  experts/layer.
- **Calibration source.** The 44.4% number uses the first 50 *decode* tokens
  as the calibration window. Calibrating on prefill routing (the preferred
  implementation) should be at least as good but is not measured here;
  prefill routing is not in the existing traces (`GGML_MMID_DUMP` only fires
  on the decode path).
- **SLRU vs LRU.** The measured rates are plain LRU / static top-k. The
  shipped SLRU is a different policy; the cache's own stats log must be the
  final arbiter.
- **The layer-8 gap.** Layer 8 was measured on 39 tokens only, because the
  GPU could not host a real ncmoe=9 run. Its 35% moves the 9-layer hit rate
  by ~1 point, so the decision is insensitive to it, but the implementation
  measurement must include all nine layers.
- **Numerics.** The split turns one `mul_mat_id` into two plus an `add`;
  float reassociation will flip near-ties (the same effect documented in
  finding 12). The gates are the existing kernel A/B, perplexity within 1%,
  and the needle test — not token identity.
- **Overlap.** The 45-50 tok/s ceiling needs a second CUDA stream, an event
  on the D2H copy of `cur`, and a scheduler that does not serialize the CPU
  op against the GPU op. Without it the expected result is 38-39 tok/s.

## 8. Reproduction

```bash
# steady-state trace (ncmoe=8, 3 prompts x 300 tokens) - exists in results/
python3 -I scripts/routing_skew.py \
  --log results/quality/route_dump3.log --period 24 --split 300 \
  --cache-experts 14

# 39-token all-CPU trace (layer 8): drop the 3 warm-up triples first
grep MMIDEXP results/quality/smoke_cpumoe.log | tail -n +10 \
  > results/quality/split_moe_decode.log
python3 -I scripts/routing_skew.py \
  --log results/quality/split_moe_decode.log --period 108 --cache-experts 14
# rows 0-26 are the ncmoe=9 CPU layers; rows 3k, 3k+1 and 3k+2 are the
# gate/up/down copies of layer k (the other rows are layers 9-35).

# per-token bytes
#   (1 - h) * 9 layers * 4 slots * 12.6398 MiB = (1 - h) * 455.0 MiB
#   fill: k * 9 layers * 12.6398 MiB
```

The static top-k rows are a few lines on top of the same MMIDEXP records:
for each layer, take the k most frequent expert ids over tokens 0..W-1 and
measure their share of the 4 slots in tokens W..end.

The logs are gitignored; the commands above regenerate them (`GGML_MMID_DUMP`
is emitted by the decode fast path in `ggml-cpu.c` and prints the four routed
expert ids of every decode MoE op).
