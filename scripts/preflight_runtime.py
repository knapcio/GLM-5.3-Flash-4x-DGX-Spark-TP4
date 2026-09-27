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
    if Path(overlay).resolve() == Path('/'):
        raise RuntimeError('root overlay destination forbidden')
    for c in containers:
        if c['Name'].lstrip('/') == name:
            raise RuntimeError('container name exists: preserve it and select a new name/path')
        for m in c['Mounts']:
            mode = m.get('Mode')
            flags = set(mode.split(',')) if type(mode) is str else set()
            # Whole-root read-only telemetry sees all paths but owns no subtree.
            # Keep every preserved subtree overlap and all writable mounts strict.
            if (m.get('Type') == 'bind' and m.get('Source') == '/'
                    and m.get('RW') is False and 'ro' in flags and 'rw' not in flags):
                continue
            if m.get('Type', 'bind') == 'bind' and overlaps(m['Source'], overlay):
                raise RuntimeError('overlay overlaps a preserved container mount: '+c['Name'])


# Only acquire fields consumed by verify. Config.Env, labels and commands are
# never requested, even temporarily before redaction.
INSPECT_FORMAT = ('{"Name":{{json .Name}},"Mounts":['
    '{{range $i, $m := .Mounts}}{{if $i}},{{end}}'
    '{"Type":{{json $m.Type}},"Source":{{json $m.Source}},'
    '"Mode":{{json $m.Mode}},"RW":{{json $m.RW}}}{{end}}]}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--container', required=True)
    p.add_argument('--overlay', required=True)
    a=p.parse_args()
    if not Path(a.overlay).is_absolute():p.error('absolute overlay path required')
    # A failed daemon/auth query is never interpreted as an empty inventory.
    ids=subprocess.check_output(['docker','ps','-aq'],text=True).split()
    data=[json.loads(line) for line in subprocess.check_output(
        ['docker','inspect','--format',INSPECT_FORMAT,*ids],text=True).splitlines()] if ids else []
    if len(data)!=len(ids):raise RuntimeError('incomplete container inventory')
    verify(data,a.container,a.overlay)
    print('runtime destination preflight PASS')
if __name__=='__main__':main()
