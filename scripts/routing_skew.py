#!/usr/bin/env python3
"""Analyse a GGML_MMID_DUMP trace: how skewed is expert routing?

Input is one line per decode MoE op, each with the 4 routed expert ids:
    MMIDEXP e0 e1 e2 e3
The ops alternate over layers, so the trace is cut into periods of P ops
(one token).  P is detected by maximising the per-layer self-similarity of
successive tokens.

For each layer it reports:
  - coverage of the top-k experts (k = 1,4,8,14,16,32,64) under uniform
    routing it would be k/128
  - the hit rate of an LRU cache of E experts with history = one token
"""
import argparse
import itertools
import math
from collections import Counter


def load(path, n_experts=128):
    ops = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("MMIDEXP"):
                p = line.split()
                ops.append([int(x) for x in p[1:5]])
    return ops


def split_layers(ops, P):
    """ops[i] belongs to layer i % P (each token emits P ops in layer order)."""
    layers = [[] for _ in range(P)]
    for i, e in enumerate(ops):
        layers[i % P].extend(e)
    return layers


PERIOD_CANDIDATES = (4, 6, 8, 9, 12, 16, 18, 24, 32, 36)


def detect_period(ops, candidates=PERIOD_CANDIDATES, max_ops=4000, verbose=False):
    """Detect ops-per-token by maximising per-layer self-similarity.

    For each candidate P the cosine similarity between the expert bags of op i
    and op i+P is averaged; the repeating layer order makes the true period
    the candidate with the highest mean similarity. Candidates need at least
    four full periods of evidence (len(ops) >= 4*P); the fallback is 8.
    """
    best, best_score = 8, -1.0
    for cand in candidates:
        if len(ops) < 4 * cand:
            continue
        bags = [Counter(ops[i]) for i in range(0, min(len(ops), max_ops))]
        sims = []
        for i in range(len(bags) - cand):
            a, b = bags[i], bags[i + cand]
            dot = sum(a[e] * b[e] for e in a if e in b)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            if na and nb:
                sims.append(dot / (na * nb))
        score = sum(sims) / len(sims) if sims else 0
        if verbose:
            print(f"  period candidate {cand}: mean cosine(same layer, t vs t+1) = {score:.4f}")
        if score > best_score:
            best, best_score = cand, score
    return best


def topk_coverage(ids, k, n_experts=128):
    c = Counter(ids)
    tot = sum(c.values())
    if not tot:
        return 0.0
    return sum(v for _, v in c.most_common(k)) / tot


def lru_hit(ids, cap, n_experts=128):
    """Hit rate of an LRU cache of `cap` experts over the id sequence."""
    cache = []
    hits = 0
    for e in ids:
        if e in cache:
            hits += 1
            cache.remove(e)
            cache.append(e)
        else:
            if len(cache) >= cap:
                cache.pop(0)
            cache.append(e)
    return hits / len(ids) if ids else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/quality/route_dump.log")
    ap.add_argument("--period", type=int, default=0, help="ops per token (0 = detect)")
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--split", type=int, default=0,
                    help="tokens per prompt; analyse each block separately")
    ap.add_argument("--cache-experts", type=int, default=14,
                    help="experts cached per layer for the LRU estimate")
    args = ap.parse_args()

    ops = load(args.log, args.n_experts)
    print(f"ops: {len(ops)}")

    if args.period:
        P = args.period
    else:
        P = detect_period(ops, verbose=True)
    print(f"period (ops per token): {P}  -> {len(ops)//P} tokens")

    if args.split:
        block = args.split * P
        blocks = [ops[i:i + block] for i in range(0, len(ops) - block + 1, block)]
        for bi, blk in enumerate(blocks):
            lay = split_layers(blk, P)
            cov = [topk_coverage(ids, args.cache_experts) for ids in lay]
            lru = [lru_hit(ids, args.cache_experts) for ids in lay]
            print(f"  prompt {bi}: {len(blk)//P} tokens  top-{args.cache_experts} "
                  f"mean {sum(cov)/len(cov):5.1%} (min {min(cov):.1%} max {max(cov):.1%})"
                  f"  LRU {sum(lru)/len(lru):5.1%}")
        ops = ops[:len(blocks) * block]

    layers = split_layers(ops, P)
    ks = [1, 4, 8, 14, 16, 32, 64]
    print("\ncoverage of the top-k experts (uniform = k/128):")
    print("  layer  " + "  ".join(f"k={k:<3d}" for k in ks) + "   LRU(h=%d)" % args.cache_experts)
    pooled = []
    for li, ids in enumerate(layers):
        row = [topk_coverage(ids, k, args.n_experts) for k in ks]
        h = lru_hit(ids, args.cache_experts, args.n_experts)
        pooled.extend(ids)
        print(f"  {li:5d}  " + "  ".join(f"{v:5.1%}" for v in row) + f"     {h:5.1%}")

    print("\npooled:")
    row = [topk_coverage(pooled, k, args.n_experts) for k in ks]
    for k, v in zip(ks, row):
        print(f"  top-{k:<3d} {v:5.1%}   (uniform {k/args.n_experts:5.1%})")
    h = lru_hit(pooled, args.cache_experts, args.n_experts)
    print(f"  LRU with {args.cache_experts} experts, pooled history: {h:.1%}")
    print(f"\nbreak-even for trading one GPU layer for a cache at ncmoe=9: 11.1%")
    # distinct experts actually used
    print(f"distinct experts seen: {len(set(pooled))}/{args.n_experts} over {len(pooled)//4} routed slots")


if __name__ == "__main__":
    main()
