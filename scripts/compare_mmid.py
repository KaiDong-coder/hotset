#!/usr/bin/env python3
"""Compare two test-mmid-cpu outputs line by line.

Both files start with a header line ("n_tokens ..."), followed by one float
per output element in the same order, so a positional comparison is exact.
Reports max absolute and relative error and exits non-zero if the two runs
disagree beyond the tolerance.
"""
import argparse
import sys


def load(path):
    with open(path) as f:
        header = f.readline().strip()
        vals = [float(x) for x in f if x.strip()]
    return header, vals


def compare_values(va, vb, rtol=5e-5, atol=1e-5):
    """Positional comparison of two value lists.

    A pair is outside tolerance when |x - y| > atol + rtol * max(|x|, |y|).
    Returns (max_abs, max_rel, worst_index, bad), where `bad` holds the
    (index, a, b) triples that failed and `worst_index` is the position of the
    largest absolute difference (0 for an empty input).
    """
    max_abs = 0.0
    max_rel = 0.0
    worst = 0
    for i, (x, y) in enumerate(zip(va, vb)):
        d = abs(x - y)
        if d > max_abs:
            max_abs = d
            worst = i
        scale = max(abs(x), abs(y), atol)
        max_rel = max(max_rel, d / scale)

    bad = [(i, va[i], vb[i]) for i in range(len(va))
           if abs(va[i] - vb[i]) > atol + rtol * max(abs(va[i]), abs(vb[i]))]
    return max_abs, max_rel, worst, bad


def compare_files(path_a, path_b, rtol=5e-5, atol=1e-5):
    """Compare two dump files; return 0 when they agree within tolerance, else 1."""
    ha, va = load(path_a)
    hb, vb = load(path_b)
    if ha != hb:
        print(f"header mismatch:\n  {ha}\n  {hb}")
        return 1
    if len(va) != len(vb):
        print(f"length mismatch: {len(va)} vs {len(vb)}")
        return 1

    max_abs, max_rel, worst, bad = compare_values(va, vb, rtol, atol)
    worst_text = (f"idx {worst}: {va[worst]:.9g} vs {vb[worst]:.9g}"
                  if va else "n/a")
    print(f"{path_a} vs {path_b}: {len(va)} values, max abs {max_abs:.3g} "
          f"({worst_text}), max rel {max_rel:.3g}")
    if bad:
        print(f"  {len(bad)} values outside tol; first 5:")
        for i, x, y in bad[:5]:
            print(f"    [{i}] {x:.9g} vs {y:.9g}")
        return 1
    print("  OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rtol", type=float, default=5e-5)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args(argv)
    return compare_files(args.a, args.b, args.rtol, args.atol)


if __name__ == "__main__":
    sys.exit(main())
