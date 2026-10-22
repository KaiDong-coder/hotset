#!/usr/bin/env python3
"""Pick llama-server placement for a MoE GGUF from what the machine can give.

Reads the GGUF tensor table directly, so the answer is model-exact: expert
tensors (blk.N.ffn_*_exps.weight) are what moves to the CPU, everything else
stays on the GPU.  Then:

  vram = non_expert + experts(layers on GPU) + kv(ctx, dtype) + buffers <= VRAM - margin
  ram  = experts(layers on CPU) + 6 GiB                                  <= MemAvailable

Printing the chosen config explains which constraint bound it.  --run execs
llama-server with the derived flags.
"""
import argparse
import ctypes
import json
import os
import shutil
import struct
import subprocess
import sys

MIB = 1024 * 1024
GIB = 1024 * MIB

# ggml type id -> (bytes per block, elements per block)
TYPE_SIZES = {
    0: (4, 1),     # F32
    1: (2, 1),     # F16
    2: (18, 32),   # Q4_0
    3: (20, 32),   # Q4_1
    6: (22, 32),   # Q5_0
    7: (24, 32),   # Q5_1
    8: (34, 32),   # Q8_0
    9: (40, 32),   # Q8_1
    10: (84, 256), # Q2_K
    11: (110, 256),# Q3_K
    12: (144, 256),# Q4_K
    13: (176, 256),# Q5_K
    14: (210, 256),# Q6_K
    15: (292, 256),# Q8_K
    16: (66, 256), # IQ2_XXS
    17: (74, 256), # IQ2_XS
    18: (98, 256), # IQ3_XXS
    19: (50, 256), # IQ1_S
    20: (18, 32),  # IQ4_NL
    21: (110, 256),# IQ3_S
    22: (82, 256),  # IQ2_S
    23: (34, 256), # IQ4_XS
    24: (1, 1),    # I8
    25: (2, 1),    # I16
    26: (4, 1),    # I32
    27: (8, 1),    # I64
    28: (8, 1),    # F64
    29: (56, 256), # IQ1_M
    30: (2, 1),    # BF16
    34: (54, 256), # TQ1_0
    35: (66, 256), # TQ2_0
    39: (17, 32),  # MXFP4
}

GGUF_MAGIC = 0x46554747


def read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def read_value(f, t):
    if t == 8:
        return read_string(f)
    if t == 9:
        (sub,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, sub) for _ in range(n)]
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def read_gguf(path):
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != GGUF_MAGIC:
            raise SystemExit(f"{path}: not a GGUF file")
        meta = {}
        for _ in range(n_kv):
            key = read_string(f)
            (t,) = struct.unpack("<I", f.read(4))
            meta[key] = read_value(f, t)
        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            (ttype,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, dims, ttype))
    return meta, tensors


def tensor_bytes(dims, ttype):
    n = 1
    for d in dims:
        n *= d
    if ttype not in TYPE_SIZES:
        raise SystemExit(f"unknown ggml type {ttype}; add it to TYPE_SIZES")
    blk_b, blk_n = TYPE_SIZES[ttype]
    return n // blk_n * blk_b


def host_mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def gpu_info():
    q = "memory.total,memory.free,pcie.link.gen.max,pcie.link.width.max,name"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None
    line = out.strip().splitlines()[0]
    total, free, gen, width, name = [x.strip() for x in line.split(",")]
    return {"total": int(total) * MIB, "free": int(free) * MIB,
            "pcie_gen": int(gen) if gen.isdigit() else None,
            "pcie_width": int(width) if width.isdigit() else None,
            "name": name}


def cores():
    phys = log = 0
    with open("/proc/cpuinfo") as f:
        ids = set()
        for line in f:
            if line.startswith("processor"):
                log += 1
            elif line.startswith("core id"):
                ids.add((log, line.split(":")[1].strip()))
            elif line.startswith("cpu cores") and not phys:
                phys = int(line.split(":")[1])
    return phys or len(ids) or log, log or os.cpu_count()


def pick_placement(expert, n_layer, nonexpert_vram, kv_bytes, buffers,
                   vram_budget, avail_host, host_margin):
    """Choose the smallest ncmoe whose VRAM footprint fits the budget, then
    check the CPU-resident experts against host RAM.  Pure: every hardware
    number is an argument, so it is unit testable without a 59 GiB GGUF.

    Note on the host check: ncmoe can never be *lowered* to save host RAM,
    because the smallest VRAM-fitting ncmoe was already chosen -- one layer
    less puts another expert layer on the GPU and exceeds the budget by
    construction.  Host pressure therefore has to be answered by a smaller KV
    (lower ctx or q8_0), which frees VRAM and only then allows fewer CPU
    layers.  This returns host_ok=False instead of pretending otherwise.

    Returns (ncmoe, vram_need, host_need, host_ok, notes).  Raises SystemExit
    when even putting every expert layer on the CPU does not fit in VRAM.
    """
    ncmoe = None
    vram_need = 0
    for cand in range(0, n_layer + 1):
        gpu_layers = n_layer - cand
        exp_gpu = sum(expert.get(li, 0) for li in range(gpu_layers))
        need = nonexpert_vram + exp_gpu + kv_bytes + buffers
        if need <= vram_budget:
            ncmoe = cand
            vram_need = need
            break
    if ncmoe is None:
        raise SystemExit(
            "refusing: even ncmoe=%d (every expert on the CPU) needs "
            "%.1f GiB VRAM, budget is %.1f GiB" % (
                n_layer, (nonexpert_vram + kv_bytes + buffers) / GIB,
                vram_budget / GIB))

    notes = []
    host_need = sum(expert.get(li, 0) for li in range(ncmoe)) + host_margin
    host_ok = host_need <= avail_host
    if not host_ok:
        notes.append(
            "host RAM too small: ncmoe=%d keeps %.1f GiB of experts on the CPU "
            "and needs %.1f GiB, only %.1f GiB available. ncmoe cannot go lower "
            "(that is the VRAM minimum); reduce ctx or use q8_0 KV to free VRAM "
            "first. Running anyway will thrash: expect page eviction and "
            "single-digit tok/s" % (
                ncmoe, (host_need - host_margin) / GIB, host_need / GIB,
                avail_host / GIB))
    return ncmoe, vram_need, host_need, host_ok, notes


def plan(args):
    meta, tensors = read_gguf(args.model)
    arch = meta.get("general.architecture", "?")

    per_layer = {}
    expert = {}
    total = 0
    for name, dims, ttype in tensors:
        b = tensor_bytes(dims, ttype)
        total += b
        if name.startswith("blk."):
            li = int(name.split(".")[1])
            per_layer.setdefault(li, 0)
            per_layer[li] += b
            if "_exps." in name:
                expert.setdefault(li, 0)
                expert[li] += b

    n_layer = meta.get(f"{arch}.block_count", len(per_layer))
    n_head = meta.get(f"{arch}.attention.head_count", 1)
    n_head_kv = meta.get(f"{arch}.attention.head_count_kv", n_head)
    if isinstance(n_head_kv, list):
        n_head_kv = n_head_kv[0]
    n_embd = meta.get(f"{arch}.embedding_length", 1)
    head_dim = (meta.get(f"{arch}.attention.key_length")
                or meta.get(f"{arch}.attention.value_length")
                or n_embd // n_head)
    n_expert = meta.get(f"{arch}.expert_count", 0)
    n_used = meta.get(f"{arch}.expert_used_count", 0)
    # bytes an expert layer actually streams per token: only the routed experts
    expert_share = (n_used / n_expert) if n_expert else 1.0

    expert_total = sum(expert.values())
    nonexpert = total - expert_total
    src_bytes = os.path.getsize(args.model)

    gpu = gpu_info()
    avail_host = host_mem_available()
    phys, log = cores()

    kv_elem = 2 if args.kv_dtype == "f16" else 1  # bytes per element per K and V
    per_tok = n_layer * 2 * n_head_kv * head_dim * kv_elem
    kv_bytes = per_tok * args.ctx
    buffers = args.buffers_gib * GIB
    # llama.cpp keeps a few tensors (output/norms) on the CPU when mmap is on
    # ("tensor overrides to CPU are used with mmap enabled"), so the non-expert
    # bytes it actually stages in VRAM are smaller than the file says.
    nonexpert_vram = max(0, nonexpert - args.cpu_resident_gib * GIB)
    margin = args.vram_margin_gib * GIB

    if gpu is None:
        raise SystemExit("no CUDA device found by nvidia-smi")
    vram_budget = gpu["total"] - margin

    # layer order in the file decides which experts sit on the GPU when only the
    # last ncmoe layers' FFN stay on the CPU.
    ncmoe, vram_need, host_need, host_ok, notes = pick_placement(
        expert, n_layer, nonexpert_vram, kv_bytes, buffers, vram_budget,
        avail_host, args.host_margin_gib * GIB)

    io_ceiling = args.host_bw_gbs * 1e9
    bytes_per_tok = expert_share * sum(expert.get(li, 0) for li in range(ncmoe))
    ceil_tps = io_ceiling / bytes_per_tok if bytes_per_tok else None

    if args.kv_dtype == "q8_0" and args.ctx >= 16384:
        notes.append(
            "q8_0 KV at >=16k context failed the long-context needle test on "
            "gpt-oss-120b (name/badge not recovered at 13k tokens, while f16 did); "
            "use it only to fit a context that f16 cannot, and never as a default")

    pcie = gpu["pcie_gen"] or 0
    op_offload = "auto" if pcie >= 4 else "off"

    cmd = []
    if args.bin:
        cmd = [args.bin, "-m", args.model, "-ngl", "99",
               "-c", str(args.ctx), "-ctk", args.kv_dtype, "-ctv", args.kv_dtype,
               "-fa", "auto", "--n-cpu-moe", str(ncmoe),
               "-t", str(phys), "-tb", str(log)]
        if op_offload == "off":
            cmd.append("--no-op-offload")

    report = {
        "model": args.model, "arch": arch, "layers": n_layer,
        "file_gib": round(src_bytes / GIB, 2),
        "expert_gib": round(expert_total / GIB, 2),
        "non_expert_gib": round(nonexpert / GIB, 2),
        "expert_share_pct": round(100.0 * expert_total / total, 2),
        "routed_experts": f"{n_used}/{n_expert}",
        "expert_mib_per_token": round(bytes_per_tok / MIB, 1) if bytes_per_tok else 0,
        "kv_per_token_kib": round(per_tok / 1024, 2),
        "ctx": args.ctx, "kv_dtype": args.kv_dtype,
        "ncmoe": ncmoe,
        "vram_need_gib": round(vram_need / GIB, 2),
        "vram_budget_gib": round(vram_budget / GIB, 2),
        "gpu_free_gib": round(gpu["free"] / GIB, 2),
        "host_need_gib": round(host_need / GIB, 2),
        "host_available_gib": round(avail_host / GIB, 2),
        "host_ok": host_ok,
        "threads": {"t": phys, "tb": log},
        "op_offload": op_offload,
        "host_ceiling_tok_s": round(ceil_tps, 1) if ceil_tps else None,
        "notes": notes,
        "cmd": cmd,
    }
    return report


def run_with_fallback(args, r):
    """Exec llama-server; if it dies allocating VRAM, move one layer back and retry."""
    cmd = list(r["cmd"])
    for attempt in range(4):
        print(f"[moe-fit] attempt {attempt}: ncmoe={r['ncmoe']}"
              f" (predicted {r['vram_need_gib']:.1f} GiB)", flush=True)
        rc = subprocess.call(cmd)
        if rc == 0:
            return rc
        if r["ncmoe"] + 1 >= r["layers"]:
            break
        r = plan(argparse.Namespace(**{**vars(args), "ctx": args.ctx,
                                       "_force_ncmoe": r["ncmoe"] + 1}))
        cmd = list(r["cmd"])
        print(f"[moe-fit] retrying with ncmoe={r['ncmoe']}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gpt-oss-120b-MXFP4.gguf")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--kv-dtype", default="f16", choices=["f16", "q8_0"])
    ap.add_argument("--vram-margin-gib", type=float, default=0.4)
    ap.add_argument("--host-margin-gib", type=float, default=6.0)
    ap.add_argument("--buffers-gib", type=float, default=0.7)
    ap.add_argument("--cpu-resident-gib", type=float, default=1.2,
                    help="non-expert tensors llama.cpp keeps off the GPU")
    ap.add_argument("--host-bw-gbs", type=float, default=16.5)
    ap.add_argument("--bin",
                    default="build/bin/llama-server")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    r = plan(args)
    if args.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"{r['model']}  [{r['arch']}]  {r['layers']} layers, "
              f"{r['file_gib']:.1f} GiB, experts {r['expert_gib']:.1f} GiB "
              f"({r['expert_share_pct']:.1f}%)")
        print(f"VRAM: need {r['vram_need_gib']:.1f} GiB <= {r['vram_budget_gib']:.1f} "
              f"GiB budget (free now {r['gpu_free_gib']:.1f}) "
              f"-> ncmoe {r['ncmoe']}/{r['layers']}")
        print(f"Host: need {r['host_need_gib']:.1f} GiB <= "
              f"{r['host_available_gib']:.1f} GiB available")
        print(f"KV: {r['kv_per_token_kib']:.1f} KiB/token x {r['ctx']} tokens "
              f"({r['kv_dtype']})")
        if r["host_ceiling_tok_s"]:
            print(f"Per token the CPU reads {r['expert_mib_per_token']:.1f} MiB "
              f"({r['routed_experts']} experts/layer); ceiling from host "
              f"bandwidth ({args.host_bw_gbs} GB/s): {r['host_ceiling_tok_s']:.1f} tok/s")
        for n in r["notes"]:
            print("  note:", n)
        print("threads -t %d -tb %d, op_offload %s" %
              (r["threads"]["t"], r["threads"]["tb"], r["op_offload"]))
        print("\n" + " ".join(r["cmd"]))
    if args.run:
        run_with_fallback(args, r)


if __name__ == "__main__":
    main()
