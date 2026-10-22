# Contributing

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
Upstream's `AGENTS.md` and `CONTRIBUTING.md` govern patches sent **to
upstream**; they do not govern this repository. Nothing here is proposed to
ggml-org.

## Run the tests first

```bash
./hotset test            # 79 unit tests + kernel A/B, CPU-only, no model
```

The kernel stage builds `test-mmid-cpu` and compares every MXFP4 fast path
against the generic `mul_mat_id` implementation at `n_tokens = 1, 2, 8, 33, 64`.
Both the unit and kernel jobs run in CI on every push.

## Rules that are not negotiable

1. **Numbers are evidence.** A decode/prefill number must come from
   `llama-server`'s `timings` block (or `llama-bench`), with the placement
   flags, context size, KV type and GPU named. A number derived from replaying a
   trace is a **projection** and must be labelled as one. `docs/PERF-LEVERS.md`
   is the model for how to separate the two.
2. **Never signal a process you did not start.** The box these numbers come from
   shares its GPU with an unrelated tenant. `scripts/gpu_queue.py` waits; it
   does not kill, renice, or deprioritise. Do not add such a call.
3. **Default-off stays default-off.** A feature behind a flag must, with the
   flag off, produce the upstream graph. Any kernel change needs an A/B against
   the generic path, not an in-kernel self-check (an in-kernel check compares
   against a reference using the same pointers — that is how the `ry[s]`
   indexing bug survived review).
4. **Do not move a projection into the measured table.** If `README.md` and
   `docs/PROBE-REPORT.md` disagree, the report is right.

## Reporting hardware

The most useful contribution right now is a measurement on hardware this
repository has never seen — see [`ROADMAP.md`](ROADMAP.md#hardware-coverage--the-cheapest-way-to-contribute).
Include:

```bash
./hotset fit --json
cc -O2 scripts/membw.c -o /tmp/membw -lpthread && /tmp/membw 8 10 1
nvidia-smi --query-gpu=name,memory.total --format=csv
```

## Code layout

- Upstream llama.cpp is untouched except the MXFP4 `mul_mat_id` paths in
  `ggml/src/ggml-cpu/ggml-cpu.c` (see [`NOTICE.md`](NOTICE.md)).
- `scripts/` is stdlib-only Python; keep it that way so the unit stage needs
  nothing but a Python interpreter.
- `tests/test_*.py` are discovered by `tests/run_tests.py`; a test file that is
  not named `test_*.py` never runs in the gate.

## Pull requests

Small, measured, one claim per PR. If a change alters a number in the README,
say which number, on which hardware, and paste the `timings` block. Explain
failures; an unmeasured "should be faster" is a hypothesis, not a fix.
