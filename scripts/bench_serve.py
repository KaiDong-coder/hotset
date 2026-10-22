#!/usr/bin/env python3
"""Benchmark one llama-server configuration and record tok/s + peak VRAM/RSS.

Starts llama-server, waits for /health, runs warmup + timed chat completions,
then reports the server-reported timings.  Peak VRAM is sampled with
nvidia-smi; peak RSS from /proc/<pid>/status VmHWM.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

PROMPT = (
    "You are a helpful assistant. Explain in detail how a mixture-of-experts "
    "language model routes tokens to experts, and why memory bandwidth rather "
    "than raw compute usually limits single-user decoding speed. Give concrete "
    "numbers for an RTX A6000 (768 GB/s) reading 2.5 GB of active weights per "
    "token. Then list five engineering techniques that reduce bytes moved per "
    "token, with one sentence on the trade-off of each."
)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def rss_kib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def parse_cache_stats(log_path):
    """Pull llama_moe_cache hit/miss lines printed when the context is freed."""
    import re
    out = {}
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = re.search(
                    r"llama_moe_cache: (\S+): hits = (\d+), misses = (\d+), "
                    r"hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB", line)
                if m:
                    out[m.group(1)] = {
                        "hits": int(m.group(2)),
                        "misses": int(m.group(3)),
                        "hit_rate_pct": float(m.group(4)),
                        "uploaded_mib": float(m.group(5)),
                    }
                m = re.search(r"large batches: (\d+) experts copied from the cache, ([\d.]+) MiB", line)
                if m:
                    out["large_batches"] = {"experts": int(m.group(1)), "mib": float(m.group(2))}
    except OSError:
        pass
    return out


def cpu_seconds(pid):
    """utime+stime of the whole process, in seconds."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(") ", 1)[1].split()
        # after the comm field: state is fields[0], utime is fields[11]
        ticks = int(fields[11]) + int(fields[12])
        return ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


def disk_bytes(pid):
    """Bytes this process has read from disk so far (actual IO, not page cache hits)."""
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def wait_health(port, proc, timeout):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server not healthy after {timeout}s")


def post_chat(port, prompt, n_predict, timeout=900):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/bin/llama-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ncmoe", type=int, default=0)
    ap.add_argument("--moe-cache-mib", type=int, default=0)
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--prompt-reps", type=int, default=1,
                    help="repeat the built-in prompt N times to make prefill longer")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--concurrent", type=int, default=1,
                    help="number of simultaneous requests (needs --parallel >= this)")
    ap.add_argument("--spec-draft-model", default=None)
    ap.add_argument("--spec-type", default=None)
    ap.add_argument("--spec-draft-n-max", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--startup-timeout", type=float, default=1800)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="extra environment variable for llama-server (repeatable)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    prompt = ("\n\n".join([PROMPT] * max(1, args.prompt_reps)))
    port = free_port()
    cmd = [
        args.bin, "-m", args.model,
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "-ctk", args.kv, "-ctv", args.kv,
        "-fa", "auto",
        "--host", "127.0.0.1", "--port", str(port),
        "--no-webui",
    ]
    if args.ncmoe:
        cmd += ["--n-cpu-moe", str(args.ncmoe)]
    if args.moe_cache_mib:
        cmd += ["--moe-cache-mib", str(args.moe_cache_mib)]
    if args.parallel and args.parallel > 1:
        cmd += ["--parallel", str(args.parallel)]
    if args.spec_draft_model:
        cmd += ["-md", args.spec_draft_model]
    if args.spec_type:
        cmd += ["--spec-type", args.spec_type]
    if args.spec_draft_n_max is not None:
        cmd += ["--spec-draft-n-max", str(args.spec_draft_n_max)]
    cmd += args.extra

    env = os.environ.copy()
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    # Baseline noise on the shared device
    base_mib = gpu_used_mib()
    log = open((args.out or "/tmp/bench_serve") + ".log", "w")
    log.write(" ".join(cmd) + "\n")
    log.write("env: " + json.dumps({k: env[k] for k in env if k.startswith(("GGML_", "LLAMA_PIN"))}) + "\n")
    log.flush()

    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    max_vram = base_mib or 0
    vram_stop = threading.Event()
    vram_lock = threading.Lock()

    def vram_sampler():
        nonlocal max_vram
        while not vram_stop.is_set():
            v = gpu_used_mib()
            if v is not None:
                with vram_lock:
                    max_vram = max(max_vram, v)
            vram_stop.wait(0.25)

    sampler = threading.Thread(target=vram_sampler, daemon=True)
    sampler.start()
    try:
        load_s = wait_health(port, proc, args.startup_timeout)
        print(f"loaded in {load_s:.1f}s (vram now {max_vram} MiB)", flush=True)

        from concurrent.futures import ThreadPoolExecutor

        results = []
        total = args.warmup + args.reps
        for i in range(total):
            c0 = cpu_seconds(proc.pid)
            d0 = disk_bytes(proc.pid)
            t0 = time.time()
            n_conc = max(1, args.concurrent)
            if n_conc == 1:
                resp = [post_chat(port, prompt, args.n_predict)]
            else:
                # give each concurrent request a different prefix so they do not share a KV prefix
                reqs = [prompt if k == 0 else f"[request {k}]\n{prompt}" for k in range(n_conc)]
                with ThreadPoolExecutor(max_workers=n_conc) as ex:
                    resp = list(ex.map(lambda p: post_chat(port, p, args.n_predict), reqs))
            wall = time.time() - t0
            c1 = cpu_seconds(proc.pid)
            d1 = disk_bytes(proc.pid)
            max_vram = max(max_vram, gpu_used_mib() or 0)
            tim = resp[0].get("timings", {})
            agg = sum(r.get("timings", {}).get("predicted_n", 0) for r in resp) / wall if wall > 0 else None
            rec = {
                "i": i,
                "warmup": i < args.warmup,
                "concurrent": n_conc,
                "wall_s": wall,
                "agg_tok_s": agg,
                "cpu_s": (c1 - c0) if c0 is not None and c1 is not None else None,
                "cpu_cores": ((c1 - c0) / wall) if c0 is not None and c1 is not None and wall > 0 else None,
                "disk_mib": ((d1 - d0) / 1048576.0) if d0 is not None and d1 is not None else None,
                "prompt_n": tim.get("prompt_n"),
                "prompt_per_s": tim.get("prompt_per_second"),
                "predicted_n": tim.get("predicted_n"),
                "predicted_per_s": tim.get("predicted_per_second"),
                "per_req_predicted_per_s": [r.get("timings", {}).get("predicted_per_second") for r in resp],
            }
            results.append(rec)
            print(json.dumps(rec), flush=True)

        peak_rss = rss_kib(proc.pid)
        timed = [r for r in results if not r["warmup"]]
        decode = [r["predicted_per_s"] for r in timed if r["predicted_per_s"]]
        prefill = [r["prompt_per_s"] for r in timed if r["prompt_per_s"]]
        cores = [r["cpu_cores"] for r in timed if r.get("cpu_cores") is not None]
        summary = {
            "cmd": cmd,
            "ctx": args.ctx,
            "ncmoe": args.ncmoe,
            "moe_cache_mib": args.moe_cache_mib,
            "kv": args.kv,
            "load_s": load_s,
            "base_vram_mib": base_mib,
            "max_vram_mib": max_vram,
            "peak_rss_kib": peak_rss,
            "decode_tok_s": sum(decode) / len(decode) if decode else None,
            "decode_tok_s_all": decode,
            "prefill_tok_s": sum(prefill) / len(prefill) if prefill else None,
            "cpu_cores_avg": sum(cores) / len(cores) if cores else None,
            "disk_mib_per_request": (sum(r["disk_mib"] for r in timed) / len(timed)
                                     if timed and all(r.get("disk_mib") is not None for r in timed) else None),
            "runs": results,
            "server_log": (args.out or "/tmp/bench_serve") + ".log",
        }
        print("SUMMARY " + json.dumps(summary), flush=True)
        if args.out:
            with open(args.out + ".metrics.json", "w") as f:
                json.dump(summary, f, indent=2)
    finally:
        vram_stop.set()
        sampler.join(timeout=2)
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        log.close()

        # cache stats are emitted when the context is destroyed, i.e. after SIGTERM
        cache_stats = parse_cache_stats((args.out or "/tmp/bench_serve") + ".log")
        print("CACHE " + json.dumps(cache_stats), flush=True)
        if args.out and cache_stats:
            try:
                with open(args.out + ".metrics.json") as f:
                    m = json.load(f)
                m["moe_cache_stats"] = cache_stats
                with open(args.out + ".metrics.json", "w") as f:
                    json.dump(m, f, indent=2)
            except (OSError, ValueError):
                pass


if __name__ == "__main__":
    main()
