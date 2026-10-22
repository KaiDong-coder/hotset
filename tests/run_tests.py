#!/usr/bin/env python3
"""One-shot test suite: placement maths, CPU MoE kernels, and (optionally)
an end-to-end greedy-consistency check.

  python3 tests/run_tests.py            # unit + kernel tests (no model needed)
  python3 tests/run_tests.py --e2e      # also loads gpt-oss-120b and compares
                                        # the fast kernels against the generic
                                        # path token by token (~10 min)

Stage 2 is the one that matters for the kernels: test-mmid-cpu is run twice,
once with the fast paths on and once with them off, and the two outputs are
compared elementwise.  A kernel bug that changes only some of the four expert
slots is invisible to an in-kernel self-check (it compares against a reference
using the same pointers), which is exactly how the 2026-10-21 `ry[s]` indexing
bug survived; comparing against the real generic path is what catches it.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CMAKE = os.environ.get("CMAKE", "cmake")
MODEL = os.path.join(ROOT, "models", "gpt-oss-120b-MXFP4.gguf")


def run(cmd, env=None, check=True):
    print(f"$ {' '.join(cmd)}", flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=ROOT, env=e)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode})")
    return p.returncode


def build():
    print("$ cmake --build build --target test-mmid-cpu", flush=True)
    p = subprocess.run([CMAKE, "--build", "build", "-j", "20",
                        "--target", "test-mmid-cpu"],
                       cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit("build failed")


def stage_unit():
    # discover, not a hardcoded filename: the mechanical fan-in runs this
    # script, so a test file it does not name never runs in the gate at all.
    # (test_probe_scripts.py sat unrun for exactly that reason.)
    print("\n=== unit: all tests/test_*.py ===", flush=True)
    run([PY, "-I", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])


def stage_kernel():
    print("\n=== kernel: fast path vs generic path ===", flush=True)
    build()
    bin_ = os.path.join(ROOT, "build", "bin", "test-mmid-cpu")
    fast = {"GGML_MMID_IL4": "1", "GGML_MMID_PREFILL": "1"}
    slow = {"GGML_MMID_IL4": "0", "GGML_MMID_PREFILL": "0"}
    out = os.path.join(ROOT, "results", "kernel_tests")
    os.makedirs(out, exist_ok=True)
    failed = []
    for n in (1, 2, 8, 33, 64):
        pa = os.path.join(out, f"fast_{n}.txt")
        pb = os.path.join(out, f"generic_{n}.txt")
        with open(pa, "w") as f:
            subprocess.run([bin_, str(n)], env={**os.environ, **fast},
                           stdout=f, check=True)
        with open(pb, "w") as f:
            subprocess.run([bin_, str(n)], env={**os.environ, **slow},
                           stdout=f, check=True)
        rc = subprocess.run([PY, "-I", "scripts/compare_mmid.py", pa, pb],
                            cwd=ROOT).returncode
        print(f"  n_tokens={n:3d}: {'PASS' if rc == 0 else 'FAIL'}", flush=True)
        if rc != 0:
            failed.append(n)
    if failed:
        raise SystemExit(f"kernel mismatch for n_tokens={failed}")


def perplexity(bin_, env, corpus, ctx=512, chunks=4):
    e = dict(os.environ)
    e.update(env)
    cmd = [bin_, "-m", MODEL, "-f", corpus, "-c", str(ctx),
           "--chunks", str(chunks), "-ngl", "99", "--n-cpu-moe", "8",
           "--no-op-offload", "-t", "10", "-tb", "20"]
    print(f"$ {' '.join(cmd)}  {env}", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit("llama-perplexity failed")
    for line in (p.stdout + p.stderr).splitlines():
        if "Final estimate: PPL" in line:
            return float(line.split("=")[1].split()[0])
    raise SystemExit("no PPL in llama-perplexity output")


def needle_check(name, env):
    """The model must recover the needle verbatim; a kernel bug loses it."""
    cmd = [PY, "-I", "scripts/quality_gate.py", "--name", name,
           "--prompts", "needle", "--needle-depth", "0.35", "--n-predict", "96",
           "--out", "results/quality"]
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    run(cmd)
    with open(os.path.join(ROOT, "results", "quality", f"{name}.json")) as f:
        d = json.load(f)
    text = next(r["text"] for r in d["results"] if r["tag"] == "needle")
    missing = [w for w in ("Okafor", "4471", "1997") if w not in text]
    if missing:
        print(f"  [{name}] missing {missing}: {text[:200]!r}")
        raise SystemExit(f"{name}: needle not recovered")
    print(f"  [{name}] needle recovered verbatim")


def stage_e2e():
    """Whole-model checks. Token identity is deliberately NOT asserted: two
    different float orders flip near-ties (top-2 within ~0.1 nats) and the
    continuations then diverge while both stay correct. Perplexity on a fixed
    corpus is the stable numeric gate, and the needle test covers recall.
    """
    print("\n=== end-to-end: perplexity + needle, fast vs generic ===", flush=True)
    if not os.path.exists(MODEL):
        raise SystemExit(f"missing {MODEL}")
    bin_ = os.path.join(ROOT, "third_party", "llama.cpp", "build", "bin",
                        "llama-perplexity")
    if not os.path.exists(bin_):
        raise SystemExit("build llama-perplexity first: "
                         "cmake --build build --target llama-perplexity")

    corpus = os.path.join(ROOT, "docs", "PROBE-REPORT.md")
    fast = {}
    slow = {"GGML_MMID_IL4": "0", "GGML_MMID_PREFILL": "0"}
    p_fast = perplexity(bin_, fast, corpus)
    p_slow = perplexity(bin_, slow, corpus)
    delta = abs(p_fast - p_slow) / p_slow
    print(f"  PPL fast {p_fast:.4f} vs generic {p_slow:.4f} ({delta:.3%})")
    if delta > 0.01:
        raise SystemExit("perplexity differs by more than 1%: kernel regression")
    print("  PASS (perplexity)")

    needle_check("e2e_fast", fast)
    needle_check("e2e_generic", slow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="also run the model")
    ap.add_argument("--skip-kernel", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    stage_unit()
    if not args.skip_kernel:
        stage_kernel()
    if args.e2e:
        stage_e2e()
    print("\nall selected stages passed")


if __name__ == "__main__":
    main()
