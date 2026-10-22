# Measured evidence

Every throughput figure in the README has one of these files behind it. They are
the raw output of `scripts/bench_serve.py`, not a transcription: each record
carries the exact server command, every run's `timings` block, peak VRAM, peak
RSS and disk bytes per request.

| claim in `README.md` | file |
|---|---|
| 30.6 tok/s decode / 97.8 prefill, `ncmoe=8` | `bench/ncmoe8_noopoff.metrics.json` |
| 25.4 / 81.1 at `ncmoe=9` (one more layer on the GPU) | `bench/ncmoe9_noopoff.metrics.json` |
| 25.4 / 36.0 at `ncmoe=9` with default op offload | `bench/base_ncmoe9.metrics.json` |
| 21.8 tok/s — the `ncmoe=12` thrash (122 GB of disk reads) | `bench/base_ncmoe12.metrics.json` |
| 7.7 tok/s — streaming experts to the GPU over PCIe 3.0 | `bench/gpu_stream_ncmoe9.metrics.json` |
| 24.9 / 179 at 32k ctx with q8_0 KV | `bench/ctx32k_q8_ncmoe9.metrics.json` |
| 26.6 tok/s aggregate at 2 concurrent streams | `bench/conc2.metrics.json` |
| EAGLE3 does not pay for byte-streaming MoE (24.5) | `bench/eagle3_ncmoe9.metrics.json` |
| cold start is ~3x slower; page warm-up fixes it | `warmup/*.json` |
| needle recall: f16 passes, q8_0 at 32k fails | `quality/needle_f16.json`, `quality/needle_q8.json` |
| the repaired kernels still write valid code / correct math | `quality/fix_code_math.json` |

Notes:

- Absolute paths in the recorded `cmd` arrays were rewritten to repo-relative
  (`build/bin/llama-server`, `models/...`) when these files were published.
  **No number was edited** — the throughput, memory and disk fields are exactly
  what the server reported.
- `run_tests.py` regenerates `results/kernel_tests/` locally; it is derived data
  and is not committed.
- These are single-box measurements with a shared GPU. Consecutive runs drift by
  ~7%, which is why the per-run `timings` arrays are kept: read the spread, not
  just the mean.
