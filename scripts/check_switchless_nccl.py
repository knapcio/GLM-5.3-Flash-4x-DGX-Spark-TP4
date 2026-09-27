#!/usr/bin/env python3
"""Check a named operator-provided patched NCCL artifact; no imports or GPU use."""
import argparse
import hashlib
from pathlib import Path
import re


def verify_library(path, expected):
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('SHA256 required')
    if not path.is_absolute():
        raise ValueError('absolute library path required')
    # A symlink is permitted only within the mounted directory: Docker cannot
    # resolve a link to a sibling build tree outside /opt/nccl.
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(path.parent.resolve()):
        raise ValueError('library symlink leaves the mounted directory')
    before = resolved.stat()
    h = hashlib.sha256()
    with resolved.open('rb') as f:
        header = f.read(20)
        # Little-endian, 64-bit AArch64 ELF (e_machine=183).
        if header[:6] != b'\x7fELF\x02\x01' or header[18:20] != b'\xb7\x00':
            raise ValueError('expected an AArch64 ELF library')
        h.update(header)
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('library changed during verification')
    if h.hexdigest() != expected:
        raise ValueError('library hash mismatch')
    return expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    verify_library(a.library, a.sha256)
    print('switchless NCCL artifact identity PASS (not a transport test)')


if __name__ == '__main__':
    main()
