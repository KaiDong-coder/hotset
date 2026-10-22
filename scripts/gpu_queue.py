#!/usr/bin/env python3
"""Wait for a free-VRAM budget, then run one command.

The box is shared with an unrelated tenant (``/opt/strata/engine/strata`` held
41.9 GiB of 49.1 GiB for most of 2026-10-21/22).  The policy is fixed and not
negotiable: never stop, re-prioritise, or otherwise interfere with a process
this script did not start.  The only correct move is to wait for the budget to
be free and let the benchmark run itself.

The wait uses *consecutive* samples on purpose.  A single free reading is not
enough: the resident tenant can briefly release VRAM between requests and a
benchmark started on that reading would collide with it.  With the defaults
(3 samples, 30 s apart) the budget must stay free for at least a minute.

On entry and on exit a JSON record is written under ``--log-dir`` (default
``results/gpu-queue``).  The record names every other process holding GPU
memory, its MiB, and how long the wait took, so a blocked run leaves evidence
instead of an empty log.

  # wait for 47 GiB, then run a dry nvidia-smi; exits 2 at once if impossible
  python3 -I scripts/gpu_queue.py --need-mib 47000 -- nvidia-smi

  # a benchmark that needs the card to stay free for two minutes
  python3 -I scripts/gpu_queue.py --need-mib 47700 --settle-samples 4 \
      --interval-s 30 --timeout-s 21600 -- \
      python3 -I scripts/bench_serve.py --model models/gpt-oss-120b-MXFP4.gguf \
          --warmup 3 --reps 3 --ncmoe 8 --ctx 8192 --kv f16 \
          --extra --no-op-offload --out results/bench-fixed/fast

Exit codes: 0 command ran (or the budget became free and no command was
given); 1 the budget did not settle before ``--timeout-s``; 2 the budget is
larger than the card and can never be satisfied; 3 nvidia-smi could not be
read; otherwise the command's own exit code.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_SETTLE_SAMPLES = 3
DEFAULT_INTERVAL_S = 30.0
DEFAULT_TIMEOUT_S = 6 * 3600.0  # 0 means wait forever

EXIT_OK = 0
EXIT_TIMEOUT = 1
EXIT_UNSATISFIABLE = 2
EXIT_NO_GPU = 3


# ── pure parsing / decisions (unit-tested without a GPU) ─────────────────────

def parse_gpu_memory(text):
    """Parse ``nvidia-smi --query-gpu=memory.total,memory.used,memory.free``.

    ``--format=csv,noheader,nounits`` gives one row per GPU.  A leading index
    column is tolerated (``index,total,used,free``); only the first GPU row is
    used.  Returns ``None`` when no usable row is present.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [col.strip() for col in next(csv.reader([line]))]

        def as_int(col):
            try:
                return int(col)
            except ValueError:
                return None

        values = [as_int(col) for col in cols]
        if len(values) == 4 and all(v is not None for v in values[1:]):
            total, used, free = values[1:]
        elif len(values) == 3 and all(v is not None for v in values):
            total, used, free = values
        else:
            # A row with [N/A]/[Not Supported] is not a usable reading; do not
            # let a skipped column shift the total/used/free triple.
            continue
        return {"total_mib": total, "used_mib": used, "free_mib": free}
    return None


def parse_compute_apps(text):
    """Parse ``nvidia-smi --query-compute-apps=pid,process_name,used_memory``.

    Returns one dict per process: ``pid``, ``process_name``, ``used_mib`` (None
    when the driver reports ``[N/A]``/``[Not Supported]``).  Rows without a
    numeric pid are skipped.  A comma inside a process path is preserved by
    treating the last column as the MiB value.
    """
    apps = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in next(csv.reader([line]))]
        if len(cols) < 3:
            continue
        try:
            pid = int(cols[0])
        except ValueError:
            continue
        try:
            used_mib = int(cols[-1])
        except ValueError:
            used_mib = None
        apps.append({
            "pid": pid,
            "process_name": ",".join(cols[1:-1]),
            "used_mib": used_mib,
        })
    return apps


def is_free(free_mib, need_mib):
    """True only when a reading is present and meets the budget exactly."""
    return free_mib is not None and free_mib >= need_mib


def budget_status(need_mib, total_mib):
    """Three-valued: ``possible``, ``unsatisfiable`` or ``unknown``."""
    if total_mib is None:
        return "unknown"
    if need_mib > total_mib:
        return "unsatisfiable"
    return "possible"


class SettleTracker:
    """Count consecutive samples that meet the budget.

    A single good reading is not enough; ``flickering`` -- one bad sample
    between good ones -- restarts the count, which is the whole point.
    """

    def __init__(self, need_mib, settle_samples):
        if settle_samples < 1:
            raise ValueError("settle_samples must be >= 1")
        self.need_mib = need_mib
        self.settle_samples = settle_samples
        self.streak = 0
        self.best_streak = 0
        self.samples = 0

    def feed(self, free_mib):
        self.samples += 1
        if is_free(free_mib, self.need_mib):
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0
        return self.settled

    @property
    def settled(self):
        return self.streak >= self.settle_samples


def wait_for_budget(sample_fn, need_mib, settle_samples, interval_s, timeout_s,
                    sleep_fn=time.sleep, clock=time.monotonic,
                    on_sample=None):
    """Poll ``sample_fn`` until the budget is free for N consecutive samples.

    ``sample_fn()`` returns a dict with ``free_mib`` (or ``None``).  The clock
    and sleep are injectable so the state machine is unit-tested without a GPU
    and without real time passing.  Returns a dict with ``status`` (``free`` or
    ``timeout``), ``waited_s``, ``samples_taken``, ``last_sample`` and the
    tracker instance.
    """
    tracker = SettleTracker(need_mib, settle_samples)
    started = clock()
    samples = 0
    while True:
        sample = sample_fn()
        samples += 1
        free_mib = sample.get("free_mib") if sample else None
        tracker.feed(free_mib)
        if on_sample is not None:
            on_sample(samples, sample, tracker)
        waited_s = clock() - started
        if tracker.settled:
            return {
                "status": "free",
                "waited_s": waited_s,
                "samples_taken": samples,
                "last_sample": sample,
                "tracker": tracker,
            }
        if timeout_s and waited_s >= timeout_s:
            return {
                "status": "timeout",
                "waited_s": waited_s,
                "samples_taken": samples,
                "last_sample": sample,
                "tracker": tracker,
            }
        sleep_fn(interval_s)


def record_path(log_dir, when, pid):
    """One record per invocation, named by UTC start time and pid."""
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(log_dir, f"gpu-queue-{stamp}-{pid}.json")


def new_record(*, need_mib, settle_samples, interval_s, timeout_s, started_at,
               sample=None, residents=None, status="waiting", command=None,
               waited_s=0.0, samples_taken=0, consecutive_free=0,
               returncode=None, reason=None, finished_at=None,
               log_path=None):
    """Build the evidence record.  Pure apart from the caller-supplied values."""
    return {
        "schema": "gpu-queue/1",
        "status": status,
        "pid": os.getpid(),
        "started_at": started_at,
        "finished_at": finished_at,
        "waited_s": round(waited_s, 3),
        "need_mib": need_mib,
        "settle_samples": settle_samples,
        "interval_s": interval_s,
        "timeout_s": timeout_s or None,
        "samples_taken": samples_taken,
        "consecutive_free": consecutive_free,
        "gpu": sample,
        "residents": residents or [],
        "command": command,
        "returncode": returncode,
        "reason": reason,
        "log_path": log_path,
    }


def write_record(path, record):
    """Atomically write the record so a half-written JSON is never left behind."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def format_residents(residents):
    if not residents:
        return "no compute processes"
    parts = []
    for app in residents:
        used = app.get("used_mib")
        used_s = f"{used} MiB" if used is not None else "unknown MiB"
        user = f", user {app['user']}" if app.get("user") else ""
        parts.append(f"pid {app['pid']} ({app['process_name']}) {used_s}{user}")
    return "; ".join(parts)


# ── live sampling (the only part that talks to the driver) ───────────────────

def _nvidia_smi(args, timeout=10):
    return subprocess.check_output(["nvidia-smi", *args],
                                   stderr=subprocess.DEVNULL,
                                   timeout=timeout, text=True)


def sample_gpu():
    """One memory reading, or ``None`` when nvidia-smi cannot be read."""
    try:
        return parse_gpu_memory(_nvidia_smi([
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]))
    except (OSError, subprocess.SubprocessError):
        return None


def read_residents():
    """Every process currently holding GPU memory, with pid/user/MiB."""
    try:
        apps = parse_compute_apps(_nvidia_smi([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]))
    except (OSError, subprocess.SubprocessError):
        return []
    for app in apps:
        app["user"] = _pid_user(app["pid"])
    return apps


def _pid_user(pid):
    """Username of a pid from /proc; ``None`` when it is not readable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])
                    try:
                        import pwd
                        return pwd.getpwuid(uid).pw_name
                    except (KeyError, ImportError):
                        return str(uid)
    except OSError:
        pass
    return None


# ── CLI ──────────────────────────────────────────────────────────────────────

def _iso_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _sample_line(n, sample, tracker):
    if sample is None:
        return f"gpu_queue: sample {n}: nvidia-smi unreadable (free streak 0/{tracker.settle_samples})"
    return (f"gpu_queue: sample {n}: free {sample['free_mib']} MiB, "
            f"need {tracker.need_mib} MiB "
            f"(free streak {tracker.streak}/{tracker.settle_samples})")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Wait until a VRAM budget is free for N consecutive "
                    "samples, then run a command.  Never interferes with any "
                    "process it did not start.",
        epilog="Exit codes: 0 ran/free, 1 timeout, 2 budget exceeds the card, "
               "3 nvidia-smi unreadable, otherwise the command's exit code.",
    )
    ap.add_argument("--need-mib", type=int, required=True,
                    help="VRAM budget required before the command may start")
    ap.add_argument("--settle-samples", type=int,
                    default=DEFAULT_SETTLE_SAMPLES,
                    help="consecutive free samples required (default: 3)")
    ap.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S,
                    help="seconds between samples (default: 30)")
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                    help="give up after this many seconds; 0 waits forever "
                         f"(default: {DEFAULT_TIMEOUT_S:.0f})")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "results", "gpu-queue"),
                    help="where the JSON evidence record is written")
    ap.add_argument("--record", default=None,
                    help="explicit record path (overrides --log-dir); the "
                         "queue uses this to read the final status back")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="command to run after '--'")
    args = ap.parse_args(argv)

    if args.need_mib <= 0:
        ap.error("--need-mib must be positive")
    if args.settle_samples < 1:
        ap.error("--settle-samples must be >= 1")
    if args.interval_s < 0:
        ap.error("--interval-s must be >= 0")
    if args.timeout_s < 0:
        ap.error("--timeout-s must be >= 0")

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    args.cmd = cmd
    return args


def main(argv=None):
    args = parse_args(argv)
    when = datetime.datetime.now(datetime.timezone.utc)
    log_path = args.record or record_path(args.log_dir, when, os.getpid())
    started_at = when.isoformat(timespec="seconds")

    def base(**kw):
        kw.setdefault("need_mib", args.need_mib)
        kw.setdefault("settle_samples", args.settle_samples)
        kw.setdefault("interval_s", args.interval_s)
        kw.setdefault("timeout_s", args.timeout_s)
        kw.setdefault("started_at", started_at)
        kw.setdefault("log_path", log_path)
        return new_record(**kw)

    sample = sample_gpu()
    if sample is None:
        record = base(status="error", reason="nvidia-smi produced no usable row",
                      finished_at=_iso_now())
        write_record(log_path, record)
        print("gpu_queue: REFUSING: nvidia-smi produced no usable row; "
              "cannot decide whether the budget is free", file=sys.stderr)
        return EXIT_NO_GPU

    residents = read_residents()
    free_mib = sample["free_mib"]
    decision = budget_status(args.need_mib, sample["total_mib"])
    print(f"gpu_queue: need {args.need_mib} MiB of {sample['total_mib']} MiB; "
          f"free {free_mib} MiB; resident: {format_residents(residents)}",
          flush=True)

    if decision == "unsatisfiable":
        reason = (f"need {args.need_mib} MiB > total {sample['total_mib']} MiB; "
                  "the budget can never be satisfied on this card")
        record = base(status="unsatisfiable", sample=sample,
                      residents=residents, reason=reason,
                      finished_at=_iso_now())
        write_record(log_path, record)
        print(f"gpu_queue: REFUSING: {reason}", file=sys.stderr)
        return EXIT_UNSATISFIABLE

    # Entry record: written before any waiting, so a run cut short still
    # leaves evidence of who was holding the card when it began.
    entry = base(status="waiting", sample=sample, residents=residents,
                 command=args.cmd or None)
    write_record(log_path, entry)
    print(f"gpu_queue: record {log_path}", flush=True)

    result = wait_for_budget(
        sample_gpu, args.need_mib, args.settle_samples, args.interval_s,
        args.timeout_s,
        on_sample=lambda n, s, t: print(_sample_line(n, s, t), flush=True),
    )
    tracker = result["tracker"]
    final_sample = result["last_sample"] or sample
    final_residents = read_residents()
    finished_at = _iso_now()

    if result["status"] == "timeout":
        reason = (f"budget {args.need_mib} MiB was not free for "
                  f"{args.settle_samples} consecutive samples within "
                  f"{args.timeout_s:.0f}s")
        record = base(status="timeout", sample=final_sample,
                      residents=final_residents, reason=reason,
                      waited_s=result["waited_s"],
                      samples_taken=result["samples_taken"],
                      consecutive_free=tracker.streak,
                      command=args.cmd or None, finished_at=finished_at)
        write_record(log_path, record)
        print(f"gpu_queue: BLOCKED: {reason}; record {log_path}",
              file=sys.stderr)
        return EXIT_TIMEOUT

    free_for = f"{tracker.streak} consecutive samples"
    print(f"gpu_queue: budget {args.need_mib} MiB free for {free_for} "
          f"after {result['waited_s']:.1f}s", flush=True)

    if not args.cmd:
        record = base(status="free", sample=final_sample,
                      residents=final_residents, waited_s=result["waited_s"],
                      samples_taken=result["samples_taken"],
                      consecutive_free=tracker.streak,
                      reason="budget became free; no command was given",
                      finished_at=finished_at)
        write_record(log_path, record)
        return EXIT_OK

    print(f"gpu_queue: running: {' '.join(args.cmd)}", flush=True)
    running = base(status="running", sample=final_sample,
                   residents=final_residents, waited_s=result["waited_s"],
                   samples_taken=result["samples_taken"],
                   consecutive_free=tracker.streak, command=args.cmd)
    write_record(log_path, running)

    try:
        returncode = subprocess.call(args.cmd)
    except OSError as exc:
        reason = f"could not start command: {exc}"
        record = base(status="error", sample=final_sample,
                      residents=final_residents, reason=reason,
                      waited_s=result["waited_s"],
                      samples_taken=result["samples_taken"],
                      consecutive_free=tracker.streak, command=args.cmd,
                      finished_at=_iso_now())
        write_record(log_path, record)
        print(f"gpu_queue: {reason}", file=sys.stderr)
        return 127

    record = base(status="ran", sample=final_sample,
                  residents=final_residents, waited_s=result["waited_s"],
                  samples_taken=result["samples_taken"],
                  consecutive_free=tracker.streak, command=args.cmd,
                  returncode=returncode, finished_at=_iso_now())
    write_record(log_path, record)
    print(f"gpu_queue: command exited {returncode}; record {log_path}",
          flush=True)
    if returncode < 0:
        return 128 - returncode
    return returncode


if __name__ == "__main__":
    sys.exit(main())
