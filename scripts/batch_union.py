#!/usr/bin/env python3
"""How much does continuous batching save on a byte-streaming CPU MoE?

A batched `mul_mat_id` reads each *distinct* expert's weight block once per
batch, no matter how many tokens in the batch route to it.  So the host bytes
per batch are set by the *union* of the routed experts, not by
`tokens * top_k`.  This script replays the expert ids recorded by
`GGML_MMID_DUMP` and reports, for several batch sizes, the union size per
layer, the host bytes per token and the resulting decode ceiling.

Usage:
    python3 -I scripts/batch_union.py results/quality/route_dump3.log
"""

import argparse
import statistics
import sys

# geometry of gpt-oss-120b MXFP4 (see docs/PROBE-REPORT.md)
EXPERTS_PER_LAYER = 128
EXPERT_MIB = 1617.9 / EXPERTS_PER_LAYER   # one expert, gate+up+down, 12.64 MiB
OPS_PER_LAYER = 3                          # gate / up / down share one routing
TOPK = 4

HOST_BW_GB_S = 16.5        # measured single-pass streaming read, 10 threads
ACHIEVED_GB_S = 12.4       # ncmoe=8 decode at 30.6 tok/s x 404.5 MiB


def parse_trace(path, n_layers):
    """MMIDEXP lines -> [token][layer] = frozenset(expert ids).

    One layer prints gate/up/down back to back with identical ids, so the
    trace is run-length collapsed first; what remains is one op per
    (token, CPU layer), and every `n_layers` ops is one token.
    """
    ops = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("MMIDEXP "):
                continue
            ids = line.split()[1:]
            if len(ids) != TOPK:
                continue
            ops.append(frozenset(int(x) for x in ids))
    collapsed = [op for i, op in enumerate(ops) if i == 0 or ops[i - 1] != op]
    if len(collapsed) % n_layers:
        raise SystemExit(
            f"trace has {len(collapsed)} MoE ops after collapsing, "
            f"not a multiple of {n_layers} layers/token (ncmoe)"
        )
    return [collapsed[t * n_layers:(t + 1) * n_layers]
            for t in range(len(collapsed) // n_layers)]


def union_profile(tokens, batch_size):
    """Mean union size per layer when `batch_size` consecutive tokens share a batch."""
    n_layers = len(tokens[0])
    totals = []
    for start in range(0, len(tokens) - batch_size + 1, batch_size):
        group = tokens[start:start + batch_size]
        for layer in range(n_layers):
            u = set()
            for tok in group:
                u |= tok[layer]
            totals.append(len(u))
    if not totals:
        return None
    return statistics.fmean(totals)


def cache_profile(tokens, batch_size, cache_size):
    """LRU of `cache_size` experts per layer; returns host bytes per token.

    An expert that is in the cache is computed on the GPU and its weights are
    not read from the host at all; a miss costs one 12.64 MiB read for the
    whole batch, however many tokens in the batch routed to it.
    """
    from collections import OrderedDict
    n_layers = len(tokens[0])
    totals = []
    for layer in range(n_layers):
        lru = OrderedDict()
        for start in range(0, len(tokens) - batch_size + 1, batch_size):
            group = tokens[start:start + batch_size]
            u = set()
            for tok in group:
                u |= tok[layer]
            misses = 0
            for e in sorted(u):                      # deterministic order
                if e in lru:
                    lru.move_to_end(e)
                else:
                    misses += 1
                    lru[e] = None
                    if len(lru) > cache_size:
                        lru.popitem(last=False)
            totals.append(misses * EXPERT_MIB / batch_size)
    return statistics.fmean(totals) * n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--ncmoe", type=int, default=8)
    ap.add_argument("--bw", type=float, default=ACHIEVED_GB_S,
                    help="achieved host bandwidth in GB/s")
    ap.add_argument("--cache", type=int, default=0,
                    help="experts per layer held resident on the GPU")
    ap.add_argument("--extra-layers", type=int, default=0,
                    help="CPU layers beyond the traced ones (the cache has to "
                         "be paid for out of VRAM); scales host bytes by "
                         "(n+extra)/n")
    args = ap.parse_args()

    tokens = parse_trace(args.trace, args.ncmoe)
    n_layers = len(tokens[0])
    print(f"trace: {args.trace}")
    print(f"tokens: {len(tokens)}   MoE layers on CPU: {n_layers} "
          f"(ncmoe={args.ncmoe})  top-{TOPK}")

    print()
    print(f"{'batch':>5} {'union/layer':>12} {'bytes/token':>12} "
          f"{'tok/s @12.4GB/s':>16} {'tok/s @16.5GB/s':>16} {'vs b=1':>8}")
    base = None
    for b in (1, 2, 4, 8, 16, 32):
        u = union_profile(tokens, b)
        if u is None:
            continue
        mib = n_layers * u * EXPERT_MIB / b
        tps = args.bw * 1000.0 / mib
        ceil = HOST_BW_GB_S * 1000.0 / mib
        if base is None:
            base = tps
        print(f"{b:>5} {u:>12.2f} {mib:>10.1f}MiB {tps:>16.1f} "
              f"{ceil:>16.1f} {tps / base:>7.2f}x")

    if args.cache > 0:
        vram_mib = args.cache * n_layers * EXPERT_MIB
        print()
        print(f"GPU expert cache: {args.cache}/layer x {n_layers} layers "
              f"= {vram_mib:.0f} MiB VRAM "
              f"(pays for itself by moving {(vram_mib / EXPERT_MIB / (n_layers + 1)):.1f} "
              f"layers to the CPU)")
        print(f"{'batch':>5} {'host bytes/token':>18} {'tok/s @12.4GB/s':>16} "
              f"{'vs cached-b=1':>14}")
        cbase = None
        for b in (1, 2, 4, 8, 16, 32):
            mib = cache_profile(tokens, b, args.cache) * \
                (n_layers + args.extra_layers) / n_layers
            tps = args.bw * 1000.0 / mib
            if cbase is None:
                cbase = tps
            print(f"{b:>5} {mib:>16.1f}MiB {tps:>16.1f} {tps / cbase:>13.2f}x")

    print()
    print("Interpretation: at b=1 the union is exactly top-4 per layer "
          f"({n_layers * TOPK * EXPERT_MIB:.1f} MiB/token). "
          "Batching only pays if the union grows slower than linearly.")


if __name__ == "__main__":
    sys.exit(main())
