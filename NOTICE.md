# NOTICE

hotset is a **fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**
(MIT License, `LICENSE`), pinned at upstream commit
`d8d92973c8c21bb4dd5d592f0fec26fe2cd1e654` and carrying local changes for
CPU-resident MoE serving of gpt-oss-120B. It is not a drop-in replacement for
upstream llama.cpp.

The upstream README is preserved at `docs/upstream-README.md`; upstream's own
documentation, examples and build system are unchanged unless listed here.

## Local changes

1. **MXFP4 `mul_mat_id` fast paths** — `ggml/src/ggml-cpu/ggml-cpu.c`
   (`ggml_mmid_dot_mxfp4_q8_0_x4` and friends): 4-expert interleaved decode and
   a multi-token prefill path, with activation row indexing fixed. Gated by
   `GGML_MMID_IL4` / `GGML_MMID_PREFILL` so the generic path remains the
   reference. Equivalence test: `tests/test-mmid-cpu.cpp` + `tests/run_tests.py`.
2. **Placement and benchmark tooling** — `scripts/`, `tests/test_*.py`,
   `docs/PROBE-REPORT.md`, `docs/PERF-LEVERS.md`, `docs/SPLIT-MOE.md`. These are
   new files, not upstream code.
3. **Repository root** — `README.md` describes hotset; the upstream README moved
   to `docs/upstream-README.md`. `CLAUDE.md` describes this fork's layout.

## Kernel changes and upstream contribution rules

The local kernel work is kept in this fork's history and is **not** proposed as
an upstream patch. Upstream `AGENTS.md` / `CONTRIBUTING.md` apply to anything
that would be sent to ggml-org/llama.cpp, not to this repository.

## Attribution

llama.cpp and ggml are the work of the ggml-org contributors (`AUTHORS`). The
model is `gpt-oss-120b` (OpenAI), MXFP4 GGUF from `ggml-org/gpt-oss-120b-GGUF`.
