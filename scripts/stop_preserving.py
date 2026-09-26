#!/usr/bin/env python3
"""Stop one named container and its verified warmup processes; never remove it."""
import argparse
import os
from pathlib import Path
import signal
import subprocess


def matches(argv, root, container, boot_pid, pid):
    prewarm = str(root / 'scripts/prewarm.py')
    boot = str(root / 'scripts/boot_warm.py')
    return ((len(argv) >= 4 and argv[1] == prewarm and argv[3] == container)
            or (pid == boot_pid and len(argv) >= 2 and argv[1] == boot))


def stop_aux(root, container, proc=Path('/proc')):
    try:
        boot_pid = int((root / 'cache/boot_warm.pid').read_text().strip())
    except (FileNotFoundError, ValueError):
        boot_pid = None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            argv = (entry / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not matches(argv, root, container, boot_pid, pid):
            continue
        fd = None
        try:
            # Pin process identity before reading its argv; PID recycling cannot
            # redirect the signal. Failure is reported, not replaced by pkill.
            fd = os.pidfd_open(pid)
            argv = (entry / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            if matches(argv, root, container, boot_pid, pid):
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                print(f'stopped verified auxiliary PID {pid}')
        except (ProcessLookupError, FileNotFoundError):
            pass
        finally:
            if fd is not None:
                os.close(fd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--container', required=True)
    a = p.parse_args()
    if not a.root.is_absolute() or not a.container or a.container.startswith('-'):
        p.error('absolute root and explicit container name required')
    stop_aux(a.root, a.container)
    subprocess.run(['docker', 'stop', '--time', '30', a.container], check=True)

if __name__ == '__main__':
    main()
