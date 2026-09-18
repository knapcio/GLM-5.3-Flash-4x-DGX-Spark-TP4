"""Host-side page-cache prewarm for a vLLM safetensors load (the GLM equivalent of ds41's fast_load).

python3 prewarm.py <model_dir> <container_name> [ahead=3] [threads=4]

vLLM's loader mmaps shard files in sorted order and page-faults them at ~0.35 GB/s single-threaded.
This sidecar reads the next `ahead` shards into the page cache with `threads` parallel pread streams
while the loader is busy with the current one, and drops (fadvise DONTNEED) shards the loader has
already consumed so cached pages never pile up in unified memory. Progress is read from the
container's tqdm line "Loading safetensors checkpoint shards: X% Completed | i/N". Exits when the
loader reports N/N or the container stops.
"""
import concurrent.futures, os, re, subprocess, sys, time

model_dir, ctn = sys.argv[1], sys.argv[2]
AHEAD = int(sys.argv[3]) if len(sys.argv) > 3 else 3
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
CHUNK = 16 << 20
files = sorted(f for f in os.listdir(model_dir) if f.endswith('.safetensors'))
paths = [os.path.join(model_dir, f) for f in files]
N = len(paths)
PROG = re.compile(r'shards:\s+\d+% Completed \| (\d+)/(\d+)')

def progress():
    try:
        out = subprocess.run(['docker', 'logs', '--tail', '200', ctn], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    text = (out.stdout + out.stderr).replace('\r', '\n')
    m = PROG.findall(text)
    if not m:
        return 0 if 'Loading model from scratch' in text or True else None
    return int(m[-1][0])

def alive():
    r = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', ctn], capture_output=True, text=True)
    return r.stdout.strip() == 'true'

def read_range(path, off, ln):
    fd = os.open(path, os.O_RDONLY)
    try:
        pos = off
        while pos < off + ln:
            b = os.pread(fd, min(CHUNK, off + ln - pos), pos)
            if not b: break
            pos += len(b)
    finally:
        os.close(fd)

def warm(path, ex):
    sz = os.path.getsize(path)
    per = (sz + THREADS - 1) // THREADS
    futs = [ex.submit(read_range, path, i * per, min(per, sz - i * per)) for i in range(THREADS) if i * per < sz]
    for f in futs: f.result()

def drop(path):
    try:
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    except Exception: pass

t0 = time.time(); warmed = -1; dropped = -1
with concurrent.futures.ThreadPoolExecutor(THREADS) as ex:
    while True:
        if not alive():
            print('container stopped', flush=True); break
        p = progress()  # shards fully consumed
        if p is None: time.sleep(1); continue
        if p >= N: break
        # drop consumed shards (keep the one being read)
        while dropped < p - 2:
            dropped += 1; drop(paths[dropped])
        target = min(N - 1, p + AHEAD)
        if warmed < target:
            warmed += 1; ts = time.time(); warm(paths[warmed], ex)
            print(f'{time.time()-t0:6.1f}s warmed {files[warmed]} in {time.time()-ts:.1f}s (loader at {p}/{N})', flush=True)
        else:
            time.sleep(0.5)
print(f'done in {time.time()-t0:.0f}s', flush=True)
