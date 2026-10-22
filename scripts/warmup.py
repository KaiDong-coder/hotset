#!/usr/bin/env python3
"""Prefault the CPU-resident MoE experts at load and hide the cold decode.

PROBE-017 showed the first request after load reads ~6.2 GiB from disk and
decodes at 8.5 tok/s, while request 4 reaches ~24.6 tok/s with ~140 MiB of
reads.  The CPU expert weights are mmapped and never prefaulted, so the first
user pays for paging 12.6 GiB (ncmoe=8) in through random 4 KiB faults.

This launcher runs one hidden warmup decode *after* the server is healthy and
*before* the first real request, so the pages the first request needs are
already resident; it then reports the first real request's decode rate and
/proc/<pid>/io read_bytes.

  # the measured default: MADV_WILLNEED on the expert ranges plus one hidden
  # decode (same prompt / length as the benchmark request), then measure the
  # first real request
  python3 scripts/warmup.py --name warm

  # cold control (no warmup, drop the expert ranges from the page cache first)
  python3 scripts/warmup.py --name ctl --prefault none \
      --hidden-warmup-tokens 0 --drop-cache

  # where the box/cgroup can hold all 12.6 GiB: prefault the ranges instead
  python3 scripts/warmup.py --name pf --prefault read \
      --hidden-warmup-tokens 0 --drop-cache

The ranges are derived from the GGUF tensor table: layers 0..ncmoe-1 are the
ones `--n-cpu-moe N` keeps on the CPU ("the first N layers", common/arg.cpp),
and the expert tensors are the ones matching llama.cpp's LLM_FFN_EXPS_REGEX
(`\\.ffn_(up|down|gate|gate_up)_(ch|)exps`).

`--prefault read` (sequential pread) only works when the process can keep
~12.6 GiB of page cache: under the quay worker's host-derived memory envelope
(25% of 32 GB = 7.6 GiB) the kernel reclaims most of it, so the default is
`willneed` (POSIX_FADV_WILLNEED, asynchronous, 0.0 s) *plus* the hidden
warmup, which faults just the hot expert set (~6 GiB) and stays inside that
envelope.  Both mechanisms run before the first *real* request; see
docs/FINDING-13-WARMUP.md.
"""
import argparse
import ctypes
import json
import mmap
import os
import re
import signal
import socket
import struct
import subprocess
import time
import urllib.request

PROMPT = ("Explain in detail how a mixture-of-experts language model routes "
          "tokens to experts, and why memory bandwidth limits decoding.")

# tensor names kept on the CPU by --n-cpu-moe N (layers 0..N-1)
EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:up|down|gate|gate_up)_(?:ch)?exps\.")

# GGUF value types
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL = 4, 5, 6, 7
GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 8, 9, 10, 11, 12
# scalar format + size for the fixed-width types
GGUF_SCALAR = {
    GGUF_UINT8: ("B", 1), GGUF_INT8: ("b", 1),
    GGUF_UINT16: ("H", 2), GGUF_INT16: ("h", 2),
    GGUF_UINT32: ("I", 4), GGUF_INT32: ("i", 4),
    GGUF_FLOAT32: ("f", 4), GGUF_BOOL: ("?", 1),
    GGUF_UINT64: ("Q", 8), GGUF_INT64: ("q", 8), GGUF_FLOAT64: ("d", 8),
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(port, payload, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(port, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become healthy")


def proc_io(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, v = line.split(":")
                d[k] = int(v)
    except OSError:
        pass
    return d


def proc_status(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmSwap:", "VmHWM:")):
                    k, v = line.split(":")
                    d[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return d


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")[0], line.split()[1]
            if k in ("MemAvailable", "Cached", "SwapFree", "SwapTotal"):
                out[k] = int(v) // 1024
    return out


# ── GGUF ──────────────────────────────────────────────────────────────────────
class GGUF:
    """Minimal GGUF v2/v3 reader: tensor names, offsets, data offset."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"{path}: not a GGUF file ({magic!r})")
            self.version = struct.unpack("<I", f.read(4))[0]
            if self.version == 1:
                n_tensors, n_kv = struct.unpack("<II", f.read(8))
            elif self.version in (2, 3):
                n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            else:
                raise ValueError(f"{path}: unsupported GGUF version {self.version}")
            self.v1 = self.version == 1
            self.meta = {}
            for _ in range(n_kv):
                key = self._str(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if key in (b"general.alignment", b"general.architecture",
                           b"general.name"):
                    self.meta[key.decode()] = self._value(f, vtype)
                else:
                    self._skip(f, vtype)
            self.tensors = []
            for _ in range(n_tensors):
                name = self._str(f).decode(errors="replace")
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                ttype = struct.unpack("<I", f.read(4))[0]
                offs = struct.unpack("<Q", f.read(8))[0]
                self.tensors.append({"name": name, "dims": dims,
                                     "type": ttype, "offs": offs})
            align = int(self.meta.get("general.alignment", 32))
            pos = f.tell()
            self.data_offset = (pos + align - 1) // align * align
        self.file_size = os.path.getsize(path)
        # tensor byte ranges, ordered by file position (pads included)
        ordered = sorted(self.tensors, key=lambda t: t["offs"])
        for i, t in enumerate(ordered):
            start = self.data_offset + t["offs"]
            end = (self.data_offset + ordered[i + 1]["offs"]
                   if i + 1 < len(ordered) else self.file_size)
            t["start"], t["end"] = start, end

    def _count(self, f):
        n = struct.unpack("<I", f.read(4))[0]
        if not self.v1:
            hi = struct.unpack("<I", f.read(4))[0]
            n |= hi << 32
        return n

    def _str(self, f):
        n = self._count(f)
        return f.read(n)

    def _value(self, f, t):
        if t in GGUF_SCALAR:
            fmt, size = GGUF_SCALAR[t]
            return struct.unpack("<" + fmt, f.read(size))[0]
        if t == GGUF_STRING:
            return self._str(f).decode(errors="replace")
        if t == GGUF_ARRAY:
            atype = struct.unpack("<I", f.read(4))[0]
            return [self._value(f, atype) for _ in range(self._count(f))]
        raise ValueError(f"unknown GGUF value type {t}")

    def _skip(self, f, t):
        if t in GGUF_SCALAR:
            f.seek(GGUF_SCALAR[t][1], 1)
        elif t == GGUF_STRING:
            self._str(f)
        elif t == GGUF_ARRAY:
            atype = struct.unpack("<I", f.read(4))[0]
            n = self._count(f)
            if atype in GGUF_SCALAR:
                f.seek(GGUF_SCALAR[atype][1] * n, 1)
            else:
                for _ in range(n):
                    self._skip(f, atype)
        else:
            raise ValueError(f"unknown GGUF value type {t}")


def cpu_expert_ranges(gguf, ncmoe):
    """Merge the file ranges of the expert tensors of the first N layers."""
    picked = []
    for t in gguf.tensors:
        m = EXPERT_RE.match(t["name"])
        if m and int(m.group(1)) < ncmoe:
            picked.append(t)
    ranges = sorted((t["start"], t["end"]) for t in picked)
    page = os.sysconf("SC_PAGE_SIZE")
    merged = []
    for a, b in ranges:
        a = a // page * page
        b = (b + page - 1) // page * page
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    total = sum(b - a for a, b in merged)
    return picked, merged, total


# ── page cache ────────────────────────────────────────────────────────────────
def fadvise(path, ranges, advice):
    fd = os.open(path, os.O_RDONLY)
    try:
        for a, b in ranges:
            os.posix_fadvise(fd, a, b - a, advice)
    finally:
        os.close(fd)


def complement_ranges(ranges, file_size, page):
    """[0, file_size) minus `ranges`, page aligned."""
    out = []
    cur = 0
    for a, b in ranges:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < file_size:
        out.append((cur, file_size // page * page))
    return out


class PinnedRanges:
    """pread the ranges into the page cache, then keep them mapped.

    The mapping is what keeps the pages alive on a 30 GB box: after loading,
    ~16 GiB of the 59 GiB model file sits in the page cache even though those
    bytes already live in VRAM, so a cache-only prefault is reclaimed before
    the first request.  MAP_PRIVATE (ACCESS_COPY) mapping without writes
    shares the page-cache pages with the server's mmap.
    """

    def __init__(self, path, ranges):
        self.path = path
        self.ranges = ranges
        self.fd = None
        self.maps = []
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)

    def pin(self):
        self.fd = os.open(self.path, os.O_RDONLY)
        for a, b in self.ranges:
            self.maps.append(mmap.mmap(self.fd, b - a,
                                       access=mmap.ACCESS_COPY, offset=a))

    def residency(self):
        """Resident bytes per mincore over the held mappings."""
        if not self.maps:
            return 0, 0
        total = resident = 0
        for m in self.maps:
            size = len(m)
            n = (size + 4095) // 4096
            vec = (ctypes.c_ubyte * n)()
            addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
            rc = self._libc.mincore(ctypes.c_void_p(addr),
                                    ctypes.c_size_t(size), vec)
            if rc != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))
            resident += sum(1 for b in vec if b & 1) * 4096
            total += size
        return min(resident, total), total

    def close(self):
        for m in self.maps:
            m.close()
        self.maps = []
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def read_ranges(path, ranges, chunk=8 << 20, progress=True):
    """Sequentially read every range; returns (bytes, seconds)."""
    fd = os.open(path, os.O_RDONLY)
    total = 0
    t0 = time.time()
    try:
        for i, (a, b) in enumerate(ranges):
            off = a
            while off < b:
                n = min(chunk, b - off)
                buf = os.pread(fd, n, off)
                if not buf:
                    raise IOError(f"short read at {off} of {path}")
                total += len(buf)
                off += len(buf)
            if progress:
                elapsed = time.time() - t0
                mib = total / 2 ** 20
                print(f"  prefault range {i + 1}/{len(ranges)}: {mib:.0f} MiB "
                      f"in {elapsed:.1f}s ({mib / max(elapsed, 1e-9):.0f} MiB/s)",
                      flush=True)
    finally:
        os.close(fd)
    return total, time.time() - t0


def prefault(path, ranges, mode, hold, pinned):
    """Fill the page cache for `ranges`. Returns a timing dict.

    With `hold`, the ranges stay mapped (PinnedRanges) so the kernel cannot
    quietly reclaim them while the first request runs.
    """
    rec = {"mode": mode, "bytes": sum(b - a for a, b in ranges)}
    if mode == "none" or not ranges:
        rec["seconds"] = 0.0
        return rec
    if mode == "willneed":
        t0 = time.time()
        fadvise(path, ranges, os.POSIX_FADV_WILLNEED)
        rec["seconds"] = time.time() - t0
    elif mode == "read":
        n, s = read_ranges(path, ranges)
        rec["seconds"], rec["read_bytes"] = s, n
        if hold:
            t0 = time.time()
            pinned.pin()
            rec["pin_seconds"] = time.time() - t0
            resident, total = pinned.residency()
            rec["resident_bytes"] = resident
            rec["resident_ratio"] = resident / total if total else None
    else:
        raise ValueError(f"unknown prefault mode {mode!r}")
    return rec


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/bin/llama-server")
    ap.add_argument("--model", default="models/gpt-oss-120b-MXFP4.gguf")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ncmoe", type=int, default=8)
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prefault", default="willneed",
                    choices=["none", "read", "willneed"],
                    help="page-cache prefault of the CPU expert ranges: "
                         "willneed (default, asynchronous MADV_WILLNEED), "
                         "none, or read (deterministic pread; wants ~12.6 GiB "
                         "of page cache)")
    ap.add_argument("--hidden-warmup-tokens", type=int, default=None,
                    help="tokens decoded once, before the measured requests, "
                         "with the same prompt; defaults to --n-predict so the "
                         "measured request reuses the same expert set "
                         "(0 disables)")
    ap.add_argument("--drop-cache", action="store_true",
                    help="DONTNEED the CPU expert ranges before launching the "
                         "server, so the run starts cold (stop other servers first)")
    ap.add_argument("--drop-gpu-cache", dest="drop_gpu_cache",
                    action="store_true", default=True,
                    help="after load, DONTNEED the file pages that are not CPU "
                         "experts: they are already in VRAM and would otherwise "
                         "evict the prefaulted expert pages")
    ap.add_argument("--keep-gpu-cache", dest="drop_gpu_cache",
                    action="store_false")
    ap.add_argument("--hold-mapping", dest="hold_mapping",
                    action="store_true", default=True,
                    help="keep the prefaulted ranges mapped so they stay "
                         "resident (mincore-verified)")
    ap.add_argument("--no-hold-mapping", dest="hold_mapping",
                    action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tensor ranges and exit")
    ap.add_argument("--name", default="warmup")
    ap.add_argument("--out", default="results/warmup")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    if args.hidden_warmup_tokens is None:
        args.hidden_warmup_tokens = args.n_predict

    g = GGUF(args.model)
    picked, ranges, total = cpu_expert_ranges(g, args.ncmoe)
    print(f"{os.path.basename(args.model)}: {len(g.tensors)} tensors, "
          f"GGUF v{g.version}, data at {g.data_offset}, "
          f"arch {g.meta.get('general.architecture')}")
    print(f"CPU expert tensors (layers 0..{args.ncmoe - 1}): {len(picked)}, "
          f"{total / 2 ** 20:.1f} MiB in {len(ranges)} ranges")
    for t in picked:
        print(f"  {t['name']:<32} {t['start']:>14} .. {t['end']:>14} "
              f"({(t['end'] - t['start']) / 2 ** 20:8.1f} MiB)")
    if args.dry_run:
        return

    os.makedirs(args.out, exist_ok=True)
    log_path = f"{args.out}/{args.name}.log"
    rec = {
        "name": args.name, "model": args.model, "bin": args.bin,
        "ncmoe": args.ncmoe, "ctx": args.ctx, "kv": args.kv,
        "prefault": args.prefault,
        "hidden_warmup_tokens": args.hidden_warmup_tokens,
        "cpu_expert_tensors": len(picked),
        "cpu_expert_ranges": len(ranges),
        "cpu_expert_bytes": total,
    }

    if args.drop_cache:
        t0 = time.time()
        fadvise(args.model, ranges, os.POSIX_FADV_DONTNEED)
        rec["drop_cache_s"] = time.time() - t0
        print(f"dropped CPU expert ranges from the page cache in "
              f"{rec['drop_cache_s']:.1f}s")

    port = free_port()
    cmd = [args.bin, "-m", args.model, "-ngl", "99", "-c", str(args.ctx),
           "-ctk", args.kv, "-ctv", args.kv, "-fa", "auto",
           "--host", "127.0.0.1", "--port", str(port), "--no-webui",
           "--n-cpu-moe", str(args.ncmoe), "--parallel", str(args.parallel),
           "--no-op-offload", "-t", "10", "-tb", "20"] + args.extra
    log_f = open(log_path, "w")
    print(" ".join(cmd), file=log_f, flush=True)
    pinned = PinnedRanges(args.model, ranges)
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        load_s = wait_healthy(port, proc)
        st = proc_status(proc.pid)
        mi = meminfo()
        print(f"loaded in {load_s:.0f}s: RSS {st.get('VmRSS')} MiB, "
              f"swap {st.get('VmSwap')} MiB, "
              f"MemAvailable {mi['MemAvailable']} MiB, Cached {mi['Cached']} MiB")
        rec["load_s"] = load_s
        rec["mem_after_load"] = mi

        if args.drop_gpu_cache:
            page = os.sysconf("SC_PAGE_SIZE")
            rest = complement_ranges(ranges, g.file_size, page)
            fadvise(args.model, rest, os.POSIX_FADV_DONTNEED)
            mi = meminfo()
            rec["mem_after_drop_gpu_cache"] = mi
            print(f"dropped {sum(b - a for a, b in rest) / 2 ** 20:.0f} MiB "
                  f"of non-expert file pages: MemAvailable "
                  f"{mi['MemAvailable']} MiB, Cached {mi['Cached']} MiB")

        # prefault AFTER load (loading reads the GPU tensors and would evict
        # anything prefetched before it) and BEFORE the first request.
        pf = prefault(args.model, ranges, args.prefault, args.hold_mapping,
                      pinned)
        rec["prefault_record"] = pf
        rec["load_total_s"] = load_s + pf.get("seconds", 0.0)
        print(f"prefault ({pf['mode']}): {pf['bytes'] / 2 ** 20:.0f} MiB in "
              f"{pf.get('seconds', 0.0):.1f}s"
              + (f", resident {pf.get('resident_ratio', 0) * 100:.1f}% "
                 f"({pf.get('resident_bytes', 0) / 2 ** 20:.0f} MiB)"
                 if "resident_ratio" in pf else ""))
        mi = meminfo()
        rec["mem_after_prefault"] = mi

        hidden = None
        if args.hidden_warmup_tokens > 0:
            io0 = proc_io(proc.pid)
            t0 = time.time()
            r = post(port, {"prompt": PROMPT, "temperature": 0.0,
                            "n_predict": args.hidden_warmup_tokens,
                            "cache_prompt": False})
            io1 = proc_io(proc.pid)
            tim = r.get("timings", {})
            hidden = {
                "wall_s": round(time.time() - t0, 2),
                "predicted_n": tim.get("predicted_n"),
                "decode_tok_s": round(tim["predicted_n"] /
                                      (tim["predicted_ms"] / 1000), 2),
                "read_mib": round((io1.get("read_bytes", 0) -
                                   io0.get("read_bytes", 0)) / 2 ** 20, 1),
            }
            rec["hidden_warmup"] = hidden
            print(f"hidden warmup: decode {hidden['decode_tok_s']} tok/s, "
                  f"read {hidden['read_mib']} MiB")

        runs = []
        for i in range(args.reps):
            if pinned.maps:
                res, tot = pinned.residency()
                print(f"  [expert pages resident before run {i}: "
                      f"{res / 2 ** 20:.0f}/{tot / 2 ** 20:.0f} MiB "
                      f"({res / tot * 100:.1f}%)]", flush=True)
                if i == 0:
                    rec["resident_before_first_request"] = {
                        "bytes": res, "total": tot, "ratio": res / tot}
            io0 = proc_io(proc.pid)
            t0 = time.time()
            r = post(port, {"prompt": PROMPT, "temperature": 0.0,
                            "n_predict": args.n_predict,
                            "cache_prompt": False})
            wall = time.time() - t0
            io1 = proc_io(proc.pid)
            tim = r.get("timings", {})
            st = proc_status(proc.pid)
            run = {
                "i": i, "wall_s": round(wall, 2),
                "predicted_n": tim.get("predicted_n"),
                "decode_tok_s": round(tim["predicted_n"] /
                                      (tim["predicted_ms"] / 1000), 2),
                "prefill_tok_s": round(tim["prompt_n"] /
                                       (tim["prompt_ms"] / 1000), 1),
                "read_mib": round((io1.get("read_bytes", 0) -
                                   io0.get("read_bytes", 0)) / 2 ** 20, 1),
                "rss_mib": st.get("VmRSS"), "swap_mib": st.get("VmSwap"),
            }
            runs.append(run)
            print(f"  run {i}: decode {run['decode_tok_s']} tok/s, prefill "
                  f"{run['prefill_tok_s']} tok/s, read {run['read_mib']} MiB, "
                  f"RSS {run['rss_mib']} MiB, swap {run['swap_mib']} MiB")
        rec["runs"] = runs
        rec["cmd"] = cmd
        with open(f"{args.out}/{args.name}.json", "w") as f:
            json.dump(rec, f, indent=1)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()
        pinned.close()


if __name__ == "__main__":
    main()
