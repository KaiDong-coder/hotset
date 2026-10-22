#!/usr/bin/env python3
"""The benchmarks that are waiting on free VRAM, run in order.

Every GPU measurement on this box is blocked by the tenant that holds most of
the card (`/opt/strata/engine/strata`, 41.9 of 49.1 GiB for most of
2026-10-21/22).  PROBE-023 had to stop without measuring.  Instead of watching
nvidia-smi by hand and racing the free moment, this file declares the waiting
list and hands each entry to `scripts/gpu_queue.py`, which waits for the
budget to stay free for N consecutive samples and then runs exactly one
command.  Entries run strictly sequentially: the next one is not even offered
to the queue until the previous process has exited, so two benchmarks can
never share the card.

The queue never stops, re-prioritises, or otherwise interferes with another
tenant's process.  A blocked wait leaves a JSON record under
`results/gpu-queue/` naming the resident pid and its MiB.

  python3 -I scripts/bench_queue.py --list          # what is waiting
  python3 -I scripts/bench_queue.py --dry-run       # the gpu_queue commands
  python3 -I scripts/bench_queue.py                 # wait, then run in order
  python3 -I scripts/bench_queue.py --only warm     # just one entry
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH_SERVE = os.path.join(HERE, "bench_serve.py")
WARMUP = os.path.join(HERE, "warmup.py")
GPU_QUEUE = os.path.join(HERE, "gpu_queue.py")

DEFAULT_MODEL = os.path.join(ROOT, "models", "gpt-oss-120b-MXFP4.gguf")
DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "bench-fixed")
DEFAULT_LOG_DIR = os.path.join(ROOT, "results", "gpu-queue")

# The measured ncmoe=8 placement needs ~47.7 GiB of VRAM (PROBE-023).
NEED_MIB_FIXED = 47700
# ncmoe=9 is a little smaller; the split-MoE cache pays for itself by moving a
# layer to the CPU (docs/PERF-LEVERS.md), so the budget stays in the same band.
NEED_MIB_SPLIT = 47700


def _bench_serve(model, common, out_name, out_dir, extra):
    """One bench_serve.py command; ``--out`` must precede ``--extra``."""
    return [
        sys.executable, "-I", BENCH_SERVE,
        "--model", model,
        "--warmup", "3", "--reps", "3",
        "--ctx", "8192", "--kv", "f16",
        *common,
        "--out", os.path.join(out_dir, out_name),
        "--extra", *extra,
    ]


def build_queue(model=None, out_dir=None, python=None):
    """The declarative waiting list.

    PROBE-023's fast/generic/warm table is ready to run.  The split-MoE A/B at
    cache 0/14/28 experts per layer stays disabled until PROBE-025 lands the
    ``--moe-cache N`` flag; with the flag absent those runs would only fail,
    so listing them is honest but running them is not.
    """
    model = model or os.environ.get("PROBE_MODEL") or DEFAULT_MODEL
    out_dir = out_dir or DEFAULT_OUT_DIR
    python = python or sys.executable

    def cmd_for(out_name, common=(), extra=("--no-op-offload",)):
        argv = _bench_serve(model, list(common), out_name, out_dir, list(extra))
        # _bench_serve used this module's interpreter; honour an injected one
        argv[0] = python
        return argv

    entries = [
        {
            "name": "fixed-fast",
            "task": "PROBE-023",
            "why": "fixed-kernel decode/prefill, fast paths on",
            "need_mib": NEED_MIB_FIXED,
            "enabled": True,
            "blocked_by": None,
            "cmd": cmd_for("fast", common=("--ncmoe", "8")),
        },
        {
            "name": "fixed-generic",
            "task": "PROBE-023",
            "why": "same placement with GGML_MMID_IL4/PREFILL off",
            "need_mib": NEED_MIB_FIXED,
            "enabled": True,
            "blocked_by": None,
            "cmd": cmd_for(
                "generic",
                common=("--ncmoe", "8",
                        "--env", "GGML_MMID_IL4=0",
                        "--env", "GGML_MMID_PREFILL=0"),
            ),
        },
        {
            "name": "warm",
            "task": "PROBE-023",
            "why": "first-request vs steady-state decode (Finding 13)",
            "need_mib": NEED_MIB_FIXED,
            "enabled": True,
            "blocked_by": None,
            "cmd": [
                python, "-I", WARMUP,
                "--model", model,
                "--name", "warm", "--out", out_dir,
            ],
        },
    ]

    split = [
        ("split-moe-cache-0", "0", "9", "A/B baseline: split path off"),
        ("split-moe-cache-14", "14", "10", "14 experts/layer cached"),
        ("split-moe-cache-28", "28", "10", "28 experts/layer cached"),
    ]
    for name, experts, ncmoe, why in split:
        entries.append({
            "name": name,
            "task": "PROBE-025",
            "why": why,
            "need_mib": NEED_MIB_SPLIT,
            "enabled": False,
            "blocked_by": ("PROBE-025 has not landed the split-MoE cache "
                           "flag (--moe-cache N)"),
            "cmd": cmd_for(
                name,
                common=("--ncmoe", ncmoe, "--moe-cache-mib", "0"),
                extra=("--no-op-offload", "--moe-cache", experts),
            ),
        })
    return entries


def select_entries(entries, only=None):
    """Filter by a comma-separated ``--only`` name list (order preserved)."""
    if not only:
        return list(entries)
    wanted = [name.strip() for name in only.split(",") if name.strip()]
    by_name = {e["name"]: e for e in entries}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise KeyError(f"unknown benchmark(s): {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def gpu_queue_argv(entry, python=None, gpu_queue=None, log_dir=None):
    """The exact gpu_queue.py invocation for one entry (pure)."""
    python = python or sys.executable
    gpu_queue = gpu_queue or GPU_QUEUE
    log_dir = log_dir or DEFAULT_LOG_DIR
    record = os.path.join(log_dir, entry["name"] + ".json")
    return [
        python, "-I", gpu_queue,
        "--need-mib", str(entry["need_mib"]),
        "--log-dir", log_dir,
        "--record", record,
        "--",
        *entry["cmd"],
    ]


def format_queue(entries):
    """Human-readable table used by ``--list`` (pure)."""
    lines = [
        f"{'name':<22} {'task':<10} {'need MiB':>9}  {'state':<8} why",
        "-" * 88,
    ]
    for entry in entries:
        state = "ready" if entry["enabled"] else "blocked"
        why = entry["why"]
        if entry.get("blocked_by"):
            why = f"{why} — {entry['blocked_by']}"
        lines.append(
            f"{entry['name']:<22} {entry['task']:<10} "
            f"{entry['need_mib']:>9}  {state:<8} {why}"
        )
    return "\n".join(lines)


def read_record(path):
    """The final gpu_queue record, or ``None`` when it is absent/unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_queue(entries, *, python=None, gpu_queue=None, log_dir=None,
              dry_run=False, runner=None):
    """Wait for and run each enabled entry, strictly in order.

    A benchmark's own non-zero exit code is reported and the queue continues;
    a *wait* failure (timeout, unsatisfiable budget, unreadable driver) means
    the card is unavailable, so the queue stops instead of burning another
    timeout per entry.
    """
    runner = runner or subprocess.call
    python = python or sys.executable
    gpu_queue = gpu_queue or GPU_QUEUE
    log_dir = log_dir or DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    enabled = [e for e in entries if e["enabled"]]
    for entry in entries:
        if not entry["enabled"]:
            print(f"skip {entry['name']}: blocked — {entry.get('blocked_by')}",
                  flush=True)

    failures = []
    for i, entry in enumerate(enabled, 1):
        argv = gpu_queue_argv(entry, python=python, gpu_queue=gpu_queue,
                              log_dir=log_dir)
        print(f"\n[{i}/{len(enabled)}] {entry['name']}: need "
              f"{entry['need_mib']} MiB ({entry['task']})", flush=True)
        print("$ " + " ".join(argv), flush=True)
        if dry_run:
            continue

        record_file = os.path.join(log_dir, entry["name"] + ".json")
        try:
            os.unlink(record_file)  # never mistake a stale record for this run
        except OSError:
            pass
        rc = runner(argv)
        record = read_record(record_file)
        status = (record or {}).get("status")
        if status in ("timeout", "unsatisfiable", "error"):
            print(f"{entry['name']}: the card is not available "
                  f"({status}); stopping the queue", file=sys.stderr)
            failures.append((entry["name"], rc, status))
            break
        if rc != 0:
            print(f"{entry['name']}: benchmark exited {rc}; continuing",
                  file=sys.stderr)
            failures.append((entry["name"], rc, status or "command failed"))

    if failures:
        print("\nqueue finished with failures:", file=sys.stderr)
        for name, rc, status in failures:
            print(f"  {name}: rc={rc} status={status}", file=sys.stderr)
        return 1
    if dry_run:
        print("\ndry run: nothing executed")
    else:
        print(f"\nqueue finished: {len(enabled)} benchmark(s) run in order")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the VRAM-blocked benchmarks sequentially through "
                    "scripts/gpu_queue.py.",
    )
    ap.add_argument("--list", action="store_true",
                    help="show the waiting list and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the gpu_queue.py commands without running")
    ap.add_argument("--only", default=None,
                    help="comma-separated entry names to run")
    ap.add_argument("--model", default=None, help="override the model path")
    ap.add_argument("--out-dir", default=None,
                    help="benchmark output directory")
    ap.add_argument("--log-dir", default=None,
                    help="gpu_queue evidence records")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    entries = build_queue(model=args.model, out_dir=args.out_dir)
    try:
        entries = select_entries(entries, args.only)
    except KeyError as exc:
        print(f"bench_queue: {exc}", file=sys.stderr)
        return 2
    if args.list:
        print(format_queue(entries))
        return 0
    os.makedirs(args.out_dir or DEFAULT_OUT_DIR, exist_ok=True)
    return run_queue(entries, log_dir=args.log_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
