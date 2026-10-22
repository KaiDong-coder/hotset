#!/usr/bin/env python3
"""Deterministic output comparison across llama-server configurations.

Launches one llama-server, asks a fixed set of prompts at temperature 0,
and writes every completion to JSON together with server timings.  Two runs
with different configs (KV dtype, kernels, placement) can then be diffed
text-for-text; the greedy continuation is the quality gate.

Long-context case: a filler document with a "needle" sentence at the top and
a question at the end.  If KV quantization damages long-range retrieval the
answer changes, which shows up as a text diff.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

FILLER = (
    "The archive room holds maintenance logs for the eastern pumping station. "
    "Each log records valve pressures, pump cycles and the duty roster. "
    "Readings were taken every four hours and signed by the shift engineer. "
)

NEEDLE = (
    "NEEDLE: The pressure relief valve at the eastern pumping station was "
    "replaced on 14 March 1997 by engineer Miriam Okafor, badge 4471.\n\n")

NEEDLE_PREFIX = "Read the archive notes below and answer the question at the end.\n\n"

NEEDLE_QUESTION = (
    "\nQUESTION: Who replaced the pressure relief valve, on what date, "
    "and what was their badge number? Answer in one sentence.\n")


def build_needle_prompt(filler_reps, needle_depth):
    """Build the long-context prompt with the needle at `needle_depth`.

    The filler document is split into `half = int(filler_reps * needle_depth)`
    paragraphs before the needle and the rest after it; depth 0 therefore puts
    the needle at the very start and 0.5 in the middle.
    """
    half = int(filler_reps * needle_depth)
    return (NEEDLE_PREFIX + FILLER * half + NEEDLE
            + FILLER * (filler_reps - half) + NEEDLE_QUESTION)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/bin/llama-server")
    ap.add_argument("--model", default="models/gpt-oss-120b-MXFP4.gguf")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ncmoe", type=int, default=9)
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--threads", "-t", type=int, default=10)
    ap.add_argument("--threads-batch", "-tb", type=int, default=20)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--filler-reps", type=int, default=340,
                    help="filler paragraphs in the long-context prompt (~35 tokens each)")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--name", default="run")
    ap.add_argument("--needle-depth", type=float, default=0.0,
                    help="0 = needle at the very start, 0.5 = middle of the filler")
    ap.add_argument("--prompts", default="route,code,math,needle",
                    help="comma separated subset of route,code,math,needle")
    ap.add_argument("--out", default="results/quality")
    ap.add_argument("--log")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap.parse_args(argv)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(port, path, payload, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                body = json.loads(r.read())
                if body.get("status") == "ok":
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become healthy")


def main(argv=None):
    args = parse_args(argv)

    port = free_port()
    env = dict(os.environ)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    cmd = [args.bin, "-m", args.model, "-ngl", str(args.ngl), "-c", str(args.ctx),
           "-ctk", args.kv, "-ctv", args.kv, "-fa", "auto",
           "--host", "127.0.0.1", "--port", str(port), "--no-webui",
           "--n-cpu-moe", str(args.ncmoe), "--no-op-offload",
           "-t", str(args.threads), "-tb", str(args.threads_batch)] + args.extra
    log_path = args.log or f"{args.out}/{args.name}.log"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_f = open(log_path, "w")
    print(" ".join(cmd), file=log_f, flush=True)
    print(f"env: {args.env}", file=log_f, flush=True)
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    try:
        load_s = wait_healthy(port, proc)
        print(f"[{args.name}] server up in {load_s:.1f}s")

        long_prompt = build_needle_prompt(args.filler_reps, args.needle_depth)

        prompts = [
            ("route", "In one paragraph, explain token routing in a "
                      "mixture-of-experts layer and why the top-4 experts for a "
                      "token are data dependent."),
            ("code", "Write a Python function that merges two sorted lists in "
                     "O(n) without using sorted(). Then state its complexity."),
            ("math", "A pump moves 3.5 litres per cycle at 40 cycles per minute. "
                     "How many cubic metres does it move in 6 hours? Show the steps."),
            ("needle", long_prompt),
        ]

        want = {t.strip() for t in args.prompts.split(",") if t.strip()}
        results = []
        for tag, text in prompts:
            if tag not in want:
                continue
            t0 = time.time()
            r = post(port, "/completion", {
                "prompt": text,
                "temperature": 0.0,
                "n_predict": args.n_predict,
                "top_k": 1,
                "cache_prompt": False,
                "n_probs": 3,
                "seed": 1234,
            })
            wall = time.time() - t0
            tim = r.get("timings", {})
            rec = {
                "tag": tag,
                "prompt_n": tim.get("prompt_n"),
                "predicted_n": tim.get("predicted_n"),
                "prompt_ms": tim.get("prompt_ms"),
                "predicted_ms": tim.get("predicted_ms"),
                "wall_s": round(wall, 3),
                "text": r.get("content", ""),
                "tokens": [p.get("token") for p in (r.get("completion_probabilities") or [])],
                "raw": r,
            }
            results.append(rec)
            print(f"[{args.name}] {tag}: prompt_n={rec['prompt_n']} "
                  f"predicted_n={rec['predicted_n']} "
                  f"prompt_ms={rec['prompt_ms']} wall={wall:.1f}s")

        os.makedirs(args.out, exist_ok=True)
        out_path = f"{args.out}/{args.name}.json"
        with open(out_path, "w") as f:
            json.dump({
                "name": args.name, "kv": args.kv, "ncmoe": args.ncmoe,
                "ctx": args.ctx, "cmd": cmd, "env": args.env,
                "load_s": load_s, "results": results,
            }, f, indent=1)
        print(f"[{args.name}] wrote {out_path}")
        return results
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()


if __name__ == "__main__":
    main()
