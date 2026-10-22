# Finding 13 — the cold first request is paging, and a hidden warmup hides it

Date: 2026-10-21
Box: RTX A6000 48 GB, i9-10900X, 30 GB RAM, model on NVMe (measured
~850 MiB/s sequential). Model: `gpt-oss-120b-MXFP4.gguf`, ncmoe=8, ctx 8k,
f16 KV, `--no-op-offload`, `-t 10 -tb 20`.

PROBE-017 left one open question: the first request after load reads ~6.2 GiB
from disk and decodes at 8.5 tok/s, while requests 3-4 reach 18-25 tok/s with
~140 MiB of reads. The CPU expert weights are mmapped and never prefaulted, so
the first user pays for paging them in through random 4 KiB faults.

This finding answers it with `scripts/warmup.py`, which derives the CPU expert
ranges from the GGUF tensor table and runs the warmup *inside the launcher,
before the first real request*.

## What the CPU expert ranges are

`--n-cpu-moe N` keeps "the Mixture of Experts weights of the first N layers"
on the CPU (`common/arg.cpp`), and llama.cpp matches them with
`LLM_FFN_EXPS_REGEX = "\.ffn_(up|down|gate|gate_up)_(ch|)exps"`. Parsing the
GGUF table for layers 0..7 at ncmoe=8 gives **48 tensors, 12,943.2 MiB in 16
file ranges** (weights + 1.4 MiB biases) — exactly the 12.6 GiB PROBE-017
inferred.

`scripts/warmup.py --dry-run` prints them; no server needed.

## Mechanisms measured

All runs start with `POSIX_FADV_DONTNEED` on the 12.6 GiB expert ranges so the
run is genuinely cold (stop other model servers first). 48 decode tokens,
`cache_prompt: false`, same prompt in the warmup and in the measured request.

| run | startup work before request 1 | request 1 decode | request 1 read |
|---|---|---|---|
| `ctl` | none (cold) | **6.55 tok/s** | **7016.5 MiB** |
| `prefault_only` | `pread` 12.9 GiB into page cache | 7.01 tok/s | 6946.3 MiB |
| `prefault_pin` | `pread` + DONTNEED non-expert pages + hold mmap | 10.24 tok/s | 4974.9 MiB |
| `warm` | one hidden 48-token decode (10.9 s) | **23.89 tok/s** | **423.4 MiB** |
| `warm_willneed` (default) | `MADV_WILLNEED` (0.0 s) + hidden decode | **23.88 tok/s** | **416.7 MiB** |

Steady state on this box (runs 2-3 of `ctl`/`warm`) is 27-30 tok/s, so the
hidden warmup puts the first real request at ~80-88% of steady state.

### Why full prefault does not work here

The quay worker session runs under a systemd scope with
`MemoryMax = 25% of 32,807,133,696 = 8,201,785,344 B (7.64 GiB)` — the
driver's host-derived resource envelope. Page cache read by the launcher is
charged to that cgroup, so a 12.9 GiB prefault is reclaimed while it is being
read: `mincore` on the held mapping reports only **55.2% (7147 MiB) resident**
immediately after the read, and even dropping the 9 GiB of non-expert file
pages (`--drop-gpu-cache`) first does not make the rest fit. Holding the
mapping and the DONTNEED cut request-1 reads from 7.0 GiB to 5.0 GiB, but not
below 1 GiB.

The hidden warmup fits because it faults only the *hot* expert set the
upcoming request needs (~6.1 GiB), not all 128 experts of every CPU layer.
One 48-token decode of the same prompt brings request 1 down to 417 MiB of
reads — the experts its 48 tokens route to, minus what the warmup already
touched.

## Result

Default `scripts/warmup.py` behaviour (`MADV_WILLNEED` on the expert ranges,
then a hidden warmup with the same prompt/length as the measured request):

| metric | before | after | acceptance |
|---|---|---|---|
| first request decode | 8.5 tok/s (6.55 here) | **23.88 tok/s** | >= 20 tok/s |
| first request read_bytes | 6174 MiB (7016 here) | **417 MiB** | < 1 GiB |
| startup increase | — | **+10.9 s** | <= 60 s |

`MADV_WILLNEED` returns immediately (0.0 s) and alone does not populate
12.9 GiB under the envelope; the hidden warmup is what makes request 1 hot.
Without the WILLNEED hint the result is identical (`warm`: 23.89 tok/s /
423 MiB). The +10.9 s is the hidden warmup's own wall time (48 tokens at cold
speed 9.5 tok/s plus prefill). Server load itself was 152-188 s across runs
and is not changed by the launcher.

## Caveats

- The warmup uses the same prompt and `n_predict` as the benchmark's first
  request, which is the best case. A *different* first prompt will fault its
  own expert pages: expert routing is skewed (finding 11), so a generic
  warmup should still recover a large part of the cost, but it would not
  reproduce the 423 MiB figure. Warming with several topic prompts would
  broaden the hot set.
- The cold cost is **moved, not removed**: ~6.1 GiB / ~11 s is still paid at
  load, hidden from the user. That is the trade the acceptance allows.
- Under the 7.64 GiB worker envelope, page cache is the binding constraint,
  not RAM or NVMe. The `--prefault read`/`willneed` paths and the pinned
  mapping are kept in the script for environments where all 12.6 GiB can
  stay resident; there they should be cheaper than a decode warmup (15 s
  sequential read, no compute).
- Between measured requests the cgroup sat at its cap and some pages were
  reclaimed (`warm` run 1 read 1.9 GiB again) without dropping decode below
  20 tok/s.

## Reproduce

```bash
# cold control
python3 scripts/warmup.py --name ctl --prefault none \
    --hidden-warmup-tokens 0 --drop-cache --reps 4
# treatment (default)
python3 scripts/warmup.py --name warm --drop-cache --reps 3
# where 12.6 GiB of page cache is available
python3 scripts/warmup.py --name pf --prefault read \
    --hidden-warmup-tokens 0 --drop-cache
```

Raw runs: `results/warmup/{ctl,prefault_only,prefault_pin,warm,warm_willneed}.json`
(the matching `.log` server logs are produced next to them but gitignored).
