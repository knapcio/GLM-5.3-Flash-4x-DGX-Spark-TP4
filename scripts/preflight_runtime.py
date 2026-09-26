#!/usr/bin/env python3
"""Read-only Docker inventory; refuse mutations of preserved runtime mounts."""
import argparse
import json
from pathlib import Path
import subprocess


def overlaps(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


def verify(containers, name, overlay):
    for c in containers:
        if c['Name'].lstrip('/') == name:
            raise RuntimeError('container name exists: preserve it and select a new name/path')
        for m in c['Mounts']:
            if m.get('Type', 'bind') == 'bind' and overlaps(m['Source'], overlay):
                raise RuntimeError('overlay overlaps a preserved container mount: '+c['Name'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--container', required=True)
    p.add_argument('--overlay', required=True)
    a=p.parse_args()
    if not Path(a.overlay).is_absolute():p.error('absolute overlay path required')
    # A failed daemon/auth query is never interpreted as an empty inventory.
    ids=subprocess.check_output(['docker','ps','-aq'],text=True).split()
    data=json.loads(subprocess.check_output(['docker','inspect',*ids],text=True)) if ids else []
    if len(data)!=len(ids):raise RuntimeError('incomplete container inventory')
    verify(data,a.container,a.overlay)
    print('runtime destination preflight PASS')
if __name__=='__main__':main()
