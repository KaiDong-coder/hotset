# gpt-oss-120b on one 48 GB GPU — probe report

Date: 2026-10-21
Hardware: RTX A6000 48 GB (768 GB/s), Intel i9-10900X (10C/20T, DDR4, measured
16.5 GB/s streaming read), 30 GB RAM, PCIe 3.0 x16.
Model: gpt-oss-120b MXFP4 GGUF, 63,387,346,208 B (59.02 GiB), 36 layers,
128 experts/layer, top-4, n_embd 2880, n_ff 2880, 64 heads / 8 KV heads.
Expert weight: 1617.9 MiB per layer (56.88 GiB total, 96.4% of the model).

All numbers are server-side `timings` from llama-server, 64 decode tokens,
152-token prompt, 2 timed runs after 1 warmup. Disk column is
`/proc/<pid>/io read_bytes` per request.

## Measured matrix

| config | decode tok/s | prefill tok/s | VRAM | notes |
|---|---|---|---|---|
| ncmoe=12, default offload | 21.8 | 29.2 | 41.7 GiB | **invalid**: 122 GB disk reads, swap full, thrash |
| ncmoe=9, default offload | 25.4 | 36.0 | 46.6 GiB | no disk reads after warmup |
| ncmoe=9, `--no-op-offload` | 25.4 | **81.1** | 46.6 GiB | 9.1 CPU cores busy in prefill |
| ncmoe=9, `GGML_OP_OFFLOAD_MIN_BATCH=1` | 7.7 | 33.9 | 46.6 GiB | GPU streams 455 MB/token over PCIe 3.0 |
| ncmoe=8, `--no-op-offload` | **30.6** | **97.8** | 47.7 GiB | 8 CPU layers is the VRAM floor at 8k f16 |
| ncmoe=9 + EAGLE3 (n-max 3) | 24.5 | 83.0 | 47.7 GiB | acceptance 65.1%, mean len 2.86 |
| ncmoe=9, ctx 32k, KV q8_0 | 25.4 | 123.3 | 46.4 GiB | 747-token prompt |

## Findings

1. **Host RAM is the first-order constraint, not VRAM.** With 30 GB RAM and
   12 CPU layers (19.4 GiB), the kernel evicted expert pages and the process
   read 122 GB from disk for 3 requests. Decode collapsed to 21.8 tok/s and
   prefill to 29 tok/s. At ncmoe=9 (14.6 GiB) disk reads drop to zero.
   Any auto-fit policy must size CPU layers to *available RAM*, not just VRAM.

2. **On PCIe 3.0, computing MoE on the CPU is 3.3x faster than streaming
   experts to the GPU.** 7.7 tok/s vs 25.4 tok/s at the same placement.
   Streaming moves 455 MB/token over 7.6 GB/s (unpinned) or ~60 ms even
   pinned. The crossover needs PCIe 4.0+ and/or a resident expert cache.

3. **Prefill must run on the CPU too.** llama.cpp's default offloads expert
   matmuls to the GPU during prefill (batch >= 32) and copies every expert
   per layer. `--no-op-offload` keeps prefill on the CPU: 36.0 -> 81.1 tok/s
   at ncmoe=9 (+125%), 97.8 tok/s at ncmoe=8.

4. **One more layer on the GPU is worth ~20%.** ncmoe=8 (47.7 GiB) vs ncmoe=9
   (46.6 GiB): decode 30.6 vs 25.4, prefill 97.8 vs 81.1. VRAM headroom is
   the cheapest decode win available on this box.

5. **Speculative decoding does not pay for byte-streaming CPU MoE.** EAGLE3
   accepts 65% of drafts (mean 2.86 tokens/pass), but a 4-token verify batch
   touches ~15 distinct experts per layer instead of 4, so CPU expert bytes
   grow 3.8x and cancel the gain: 24.5 vs 25.4 tok/s. Speculation only makes
   sense where weights are resident (GPU) or when the batch shares experts.

6. **The host memory system is the hard ceiling.** A 1 GB single-pass read
   with 10 threads measures 16.5 GB/s (THP and 20 threads do not help).
   Decode reads `ncmoe * 50.56 MB` per token:
   - ncmoe=8: 404 MB -> 24.5 ms -> **40.8 tok/s theoretical max**
   - ncmoe=9: 455 MB -> 27.6 ms -> 36.2 tok/s theoretical max
   The engine currently runs at 11.5 GB/s (70% of the ceiling); 30.6 tok/s is
   75% of the theoretical maximum for ncmoe=8.

7. **The CPU MXFP4 kernel leaves ~40% on the table.** A standalone kernel that
   keeps four row streams in flight per thread reaches 15.8-16.7 GB/s vs
   7.5-11.4 GB/s for llama.cpp's one-stream-per-thread MUL_MAT_ID path.
   Thread sampling shows all 10 workers at ~84% busy but only ~11.5 GB/s:
   the limit is memory-level parallelism, not instruction throughput.

8. **The interleaved decode kernel was implemented and it works, but the
   in-situ gain is 11%, not 25%.** `ggml_compute_forward_mul_mat_id` now has an
   MXFP4/Q8_0 fast path for `n_tokens == 1, n_ids == 4`: every thread streams
   four experts' rows in lockstep (`ggml_mmid_dot_mxfp4_q8_0_x4`), contiguous
   row ranges per thread. Measured per MoE op in one server run (alternating
   the two implementations call by call, so both see identical machine state):
   generic 1.236 ms/op (13.4 GB/s) vs interleaved 1.093 ms/op (15.1 GB/s),
   -11.6%, i.e. 92% of the 16.5 GB/s host ceiling. The generic path was
   already closer to the ceiling on real expert tensors than the microbench
   suggested. Remaining headroom on the kernel is ~8% of the MoE op.
   *Correction (see finding 12): the in-situ "16/16 exact" check used the same
   activation pointer as the kernel, so it could not catch the indexing bug
   fixed later. The timing A/B is unaffected — the amount of memory streamed
   is identical — but equivalence is now demonstrated against the generic
   path instead.*
   End-to-end decode moved 28.2 -> 29.2 tok/s in a quiet run; back-to-back
   runs drift by ~7% because the GPU is shared, so the op-level A/B above is
   the trustworthy number.

9. **A multi-token prefill kernel is worth 1.65x on the op and 19% end to end.**
   `ggml_mmid_gemm_r2t4` unpacks an MXFP4 weight block once and dots it against
   two weight rows x up to four token vectors, so prefill stops paying a
   dequant per (row, token) pair. Same-run A/B: 89.3 -> 54.5 ms/op (1.65x).
   End to end with `-t 10 -tb 20`: 137 -> 155-160 tok/s at `-ub 512` (131 ->
   156) and the same at the default ubatch, i.e. **ubatch size is not the
   limiter** — an earlier 64-token histogram cap had made it look that way.
   Amdahl explains the gap: CPU MoE is ~55% of prefill wall time and the other
   45% is the 28 GPU-resident layers, which the kernel does not touch. The
   300 tok/s target needs overlap or more layers on the GPU, not a faster
   kernel.

10. **q8_0 KV is not free at long context.** At 32k ctx a 13,333-token needle
    prompt: f16 (ncmoe=9) answers "Engineer Miriam Okafor, badge 4471, 14 March
    1997" and keeps repeating it; q8_0 (ncmoe=8, i.e. one more layer on the GPU)
    never recovers the name or badge in 400 tokens, it degenerates into
    "The answer: \"Engineer". Prefill 164 -> 179 tok/s and decode 24.0 -> 24.9
    in favour of q8_0, and it is the only way to fit 32k on 48 GiB with
    ncmoe=8, but the retrieval loss is a quality regression, so f16 stays the
    default and the launcher warns for q8_0 at ctx >= 16k.

11. **Expert routing is heavily skewed, and that is the next decode lever.**
    Instrumented decode op (`GGML_MMID_DUMP`) over three topics x 300 tokens:
    24 MoE ops per token = 8 CPU layers x 3 matrices (gate/up/down). Per layer,
    the top-14 experts absorb 54-68% of the routed slots (uniform would be
    10.9%), and an LRU of 14 experts/layer hits 46-54% of the time. Trading one
    resident layer for a 1.4 GiB VRAM expert cache breaks even at an 11.1% hit
    rate, so the measured skew is 4-5x above break-even. Caveat: the cache has
    to split each MoE matmul between the CPU (uncached experts) and the GPU
    (cached ones), which is a graph-level change, not a kernel tweak.

12. **The first real kernel test found a live indexing bug that the in-kernel
    self-check could not see.** `mul_mat_id` stores quantised activations in
    `params->wdata` as `(i11 + i12*ne11)` — one row per *b* column, and the MoE
    graph reshapes the input to `[n_embd, 1, n_tokens]`, so `ne11 == 1` and all
    four expert slots of a token share one activation row. Both fast paths
    indexed that buffer by the expert slot instead (`wdata + s*row_size` in
    decode, `+ (id + token)*row_size` in prefill), so slots 1-3 read stale or
    neighbouring-token data. The old "16/16 exact" self-check compared the
    kernel against a scalar reference built from the *same* pointer, which is
    why it passed. Comparing against the generic path elementwise
    (`tests/test-mmid-cpu.cpp`, `scripts/compare_mmid.py`) shows the bug
    immediately: max rel err 0.997 at 1 token, 311 at 8. After the fix:
    max rel err 1e-7..1e-6 at 1, 2, 8, 33 and 64 tokens. End to end the greedy
    token streams still diverge from the generic path (token 19 of 96 on a
    13k-token needle prompt), but at every divergence the top-2 logprobs are
    within ~0.1 nats and both runs recover the needle verbatim — that is float
    reassociation, not a semantic difference. The throughput numbers above
    stand (same bytes streamed); the lesson is that a kernel equivalence check
    has to be an A/B against the reference implementation, never a
    self-check.

13. **The cold first request is paging, and one hidden warmup decode hides
    it.** `scripts/warmup.py` parses the GGUF tensor table, identifies the
    CPU expert ranges (`--n-cpu-moe N` keeps layers 0..N-1 on the CPU; 48
    tensors, 12,943.2 MiB in 16 ranges at ncmoe=8), and runs one 48-token
    warmup decode before the first real request. Cold first request: 6.55
    tok/s / 7016 MiB read. Default (asynchronous `MADV_WILLNEED` +
    hidden warmup, 10.9 s, part of load): 23.88 tok/s / 417 MiB read, against
    27-30 tok/s steady state. A full
    12.9 GiB page-cache prefault (pread, or `MADV_WILLNEED`) does *not* work
    under the quay worker's host-derived 7.64 GiB memory envelope: mincore
    shows only 55% of the ranges resident and request 1 still reads 5.0 GiB
    even after dropping the non-expert file pages. The warmup wins because
    it faults only the hot expert set (~6.1 GiB). Caveat: the warmup uses the
    same prompt as the first request; a different first prompt still faults
    its own experts, though routing skew (finding 11) means most of the cost
    should transfer. Details and raw runs: `docs/FINDING-13-WARMUP.md`,
    `results/warmup/`.

## Derived launch policy (v0.1)

```
n_cpu_layers = smallest N such that
    non_expert + (36-N)*1617.9 MiB + KV(ctx, dtype) + 1.5 GiB  <=  VRAM
host_bytes   = N * 1617.9 MiB
require      host_bytes + 6 GiB <= MemAvailable, else raise N (or lower ctx)

if pcie_gen <= 3 and no resident expert cache:
    op_offload = off            # CPU prefill + CPU MoE
else:
    op_offload = auto           # GPU prefill, GPU-streamed decode
    speculation = on            # only in this branch
```

Quality gates: all placement modes must produce identical greedy output for
the same prompt; KV q8_0 must be checked against f16 on a task suite before
being enabled by default.

## Where the remaining decode time goes (ncmoe=8)

Per token at 29 tok/s: 34.5 ms total, of which 24 MXFP4 MoE ops x 1.09 ms =
26.2 ms (76%). The other ~8 ms is GPU-side work (36 layers of attention plus
28 layers of MoE) that is serialized after each CPU MoE op. There is no
host-bandwidth win left; the next lever is overlapping CPU MoE with GPU work,
not a faster kernel.

## Next steps

1. ~~MXFP4 CPU MoE kernel with 4-way interleaving~~ — done (finding 8):
   +11.6% per op, 92% of the host ceiling.
2. ~~Prefill kernel: tiled GEMM over experts~~ — done (finding 9):
   +19% end to end. The remaining 45% of prefill is GPU-side, so the next
   prefill lever is overlap or placement, not the kernel.
   Open: expert grouping (sort the 512 x 4 expert slots per layer so a weight
   block serves 8-16 tokens per unpack instead of 4).
3. Auto-fit: compute ncmoe from VRAM *and* `MemAvailable`, refuse configs
   that would thrash (the ncmoe=12 experiment is the failure mode).
4. Verify long-context quality at 32k with q8_0 KV.
5. Pipeline CPU MoE against GPU attention/MoE (the ~8 ms serialized tail).
   Note: within one request the chain is strictly dependent (routing needs the
   previous layer's output), so the overlap has to come from concurrent
   requests or from splitting the graph by layer.
6. Ship `scripts/moe_fit.py`: reads the GGUF tensor table, picks `ncmoe` from
   VRAM *and* MemAvailable, derives `-t`/`-tb` from the physical/logical core
   counts, and refuses thrashing configs. Reproduces ncmoe=9 at 8k f16 and
   ncmoe=9 at 32k q8_0 on this box.
