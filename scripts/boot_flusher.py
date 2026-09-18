"""Temporary reference-recipe cache release, only during this model's boot."""
import os, pathlib, signal, time
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
end = time.monotonic() + 3600
while running and time.monotonic() < end:
    os.sync()
    pathlib.Path('/proc/sys/vm/drop_caches').write_text('3\n')
    print(f'{time.time():.0f} boot cache flush', flush=True)
    for _ in range(60):
        if not running: break
        time.sleep(1)
