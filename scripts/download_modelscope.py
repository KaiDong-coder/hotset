#!/usr/bin/env python3
"""Parallel ranged download from ModelScope with resume-by-chunk."""
import os, sys, time, threading, hashlib
import requests

URL = "https://www.modelscope.cn/models/ggml-org/gpt-oss-120b-GGUF/resolve/master/gpt-oss-120b-MXFP4.gguf"
OUT = sys.argv[1] if len(sys.argv) > 1 else "models/gpt-oss-120b-MXFP4.gguf"
EXPECTED = 63_387_346_208
THREADS = 8
REQ_CHUNK = 32 * 1024 * 1024

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fd = os.open(OUT, os.O_RDWR | os.O_CREAT)
os.ftruncate(fd, EXPECTED)

# determine total size
total = EXPECTED
try:
    r = requests.head(URL, allow_redirects=True, timeout=30)
    cl = r.headers.get("Content-Length")
    if cl:
        total = int(cl)
        print(f"HEAD size={total}", flush=True)
        os.ftruncate(fd, total)
except Exception as e:
    print(f"HEAD failed: {e}; using expected {EXPECTED}", flush=True)

part = (total + THREADS - 1) // THREADS
lock = threading.Lock()
done = 0
t0 = time.time()
errors = []

def worker(idx):
    global done
    start = idx * part
    end = min(total - 1, (idx + 1) * part - 1)
    pos = start
    sess = requests.Session()
    while pos <= end:
        req_end = min(end, pos + REQ_CHUNK - 1)
        headers = {"Range": f"bytes={pos}-{req_end}"}
        for attempt in range(8):
            try:
                with sess.get(URL, headers=headers, stream=True, timeout=(20, 180), allow_redirects=True) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f"status {r.status_code} for {pos}-{req_end}")
                    got = 0
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        if not chunk:
                            continue
                        os.pwrite(fd, chunk, pos + got)
                        got += len(chunk)
                        with lock:
                            done += len(chunk)
                    if got != req_end - pos + 1:
                        raise RuntimeError(f"short read {got} != {req_end-pos+1}")
                pos = req_end + 1
                break
            except Exception as e:
                if attempt == 7:
                    errors.append(f"thread {idx} at {pos}: {e}")
                    return
                time.sleep(1.5 * (attempt + 1))
    print(f"thread {idx} done", flush=True)

def reporter():
    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(10)
        with lock:
            d = done
        speed = (d - last) / 10 / 1e6
        last = d
        pct = 100.0 * d / total
        print(f"{pct:5.1f}% {d/1e9:6.2f}/{total/1e9:.2f} GB {speed:6.1f} MB/s", flush=True)

threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(THREADS)]
for t in threads: t.start()
rep = threading.Thread(target=reporter, daemon=True); rep.start()
for t in threads: t.join()
os.fsync(fd)
os.close(fd)
print(f"download finished in {time.time()-t0:.1f}s; errors={errors}", flush=True)
if errors:
    sys.exit(2)
# size check
sz = os.path.getsize(OUT)
print(f"size={sz} expected={total} ok={sz==total}", flush=True)
if sz != total:
    sys.exit(3)
