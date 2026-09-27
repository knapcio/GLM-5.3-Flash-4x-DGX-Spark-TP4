import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TransportTests(unittest.TestCase):
    def render(self, extra='', dry=True):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); bin = p / 'bin'; bin.mkdir(); log = p / 'calls'
            for name in ('ssh', 'docker', 'rsync'):
                f = bin / name; f.write_text('#!/bin/sh\necho EXTERNAL >> "$TEST_CALLS"\nexit 97\n'); f.chmod(0o755)
            config = p / 'config'
            config.write_text('source .env.example\n' + extra + '\n')
            script = ROOT / 'start.sh'
            env = {**os.environ, 'PATH': str(bin) + ':' + os.environ['PATH'], 'ENV_FILE': str(config), 'TEST_CALLS': str(log), 'DRY': '1' if dry else '0'}
            r = subprocess.run(['bash', str(script), 'serve'], cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertFalse(log.exists(), 'test attempted a remote command')
            return r

    def switchless(self):
        return '\n'.join(['TRANSPORT=switchless', 'NCCL_HOST_DIR=/opt/test-nccl', 'SWITCHLESS_NCCL_SHA256='+'a'*64,
                          'SWITCHLESS_ADDR_RANGE=10.100.224.0/22', 'SWITCHLESS_SUBNET_PREFIX_LEN=24'])

    def commands(self, text):
        matches = re.findall(r'^\[[^\]]+\] (docker run .*?)(?=\n\[|\nlaunched|\Z)', text, re.M | re.S)
        self.assertEqual(len(matches), 4)
        return [shlex.split(s) for s in matches]

    def env(self, cmd):
        pairs = [cmd[i+1].split('=', 1) for i, a in enumerate(cmd) if a == '-e']
        self.assertEqual(len(pairs), len({p[0] for p in pairs}), 'duplicate environment keys')
        return dict(pairs)

    def test_switched_default_identical_commands(self):
        after = self.render()
        self.assertEqual(after.returncode, 0, after.stderr)
        expected = json.loads((ROOT / 'tests/fixtures/switched-commands.json').read_text())['sha256']
        actual = [hashlib.sha256(json.dumps([x.replace(str(Path.home()), '${HOME}') for x in cmd], separators=(',', ':')).encode()).hexdigest()
                  for cmd in self.commands(after.stdout)]
        self.assertEqual(actual, expected)

    def test_all_ranks_transport_only(self):
        orig = self.commands(self.render().stdout)
        r = self.render(self.switchless()); self.assertEqual(r.returncode, 0, r.stderr)
        ring = self.commands(r.stdout)
        transport = {'GLM_ROCE_ALLREDUCE', 'B12X_ROCE_HCA', 'TORCH_USE_RTLD_GLOBAL', 'GLOO_SOCKET_IFNAME', 'MN_IF_NAME', 'TP_SOCKET_IFNAME'}
        def strip(cmd):
            out = []; i = 0
            while i < len(cmd):
                if cmd[i] == '-e' and (cmd[i+1].startswith('NCCL_') or cmd[i+1].split('=')[0] in transport): i += 2; continue
                if cmd[i] == '-v' and cmd[i+1].endswith(':/opt/nccl:ro'): i += 2; continue
                out.append(cmd[i]); i += 1
            return out
        for a,b in zip(orig, ring):
            self.assertEqual(strip(a), strip(b))
            e = self.env(b)
            for k,v in {'GLM_ROCE_ALLREDUCE':'0','NCCL_ALGO':'Ring','NCCL_SWITCHLESS_RING_ONLY':'1','NCCL_CROSS_NIC':'1','NCCL_IB_SUBNET_AWARE_ROUTING':'1','NCCL_IB_EXTENDED_IPV4_GIDS':'1','NCCL_IB_PRESERVE_PCI_DOMAIN':'1','NCCL_IB_ADDR_RANGE':'10.100.224.0/22'}.items(): self.assertEqual(e[k],v)
            self.assertNotIn('B12X_ROCE_HCA',e)
            self.assertNotIn('NCCL_IB_GID_INDEX',e)
            self.assertEqual(e['LD_PRELOAD'],e['VLLM_NCCL_SO_PATH'])
        self.assertEqual(r.stdout.count('--library /opt/test-nccl/libnccl.so.2.30.7 --sha256'),4)

    def test_dual_pf_allowlist_all_four_ranks(self):
        hcas = 'rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1'
        r = self.render(self.switchless() + '\nIB_HCA=' + hcas)
        self.assertEqual(r.returncode, 0, r.stderr)
        for cmd in self.commands(r.stdout):
            e = self.env(cmd)
            self.assertEqual(e['NCCL_IB_HCA'], '=' + hcas)
            self.assertEqual(e['NCCL_IB_MERGE_NICS'], '0')
            self.assertEqual(e['NCCL_IB_EXTENDED_IPV4_GIDS'], '1')
            self.assertEqual(e['NCCL_IB_PRESERVE_PCI_DOMAIN'], '1')

    def test_bad_config_rejected_before_remote(self):
        cases = ['TRANSPORT=oops', self.switchless()+'\nSWITCHLESS_NCCL_SHA256=bad',
                 self.switchless()+'\nNCCL_HOST_DIR="/tmp/a;bad"',
                 self.switchless()+'\nSWITCHLESS_ADDR_RANGE=10.1.2.3/22',
                 self.switchless()+'\nSWITCHLESS_SUBNET_PREFIX_LEN=40',
                 self.switchless()+'\nIPS="1.1.1.1 1.1.1.1 2.2.2.2 3.3.3.3"',
                 self.switchless()+'\nHOSTS="a b"',
                 self.switchless()+'\nEXTRA_ENV="$EXTRA_ENV NCCL_ALGO=Tree"',
                 self.switchless()+'\nEXTRA_ENV="$EXTRA_ENV LD_PRELOAD=/evil"']
        for case in cases:
            with self.subTest(case=case): self.assertNotEqual(self.render(case,dry=False).returncode,0)


class ArtifactTests(unittest.TestCase):
    def test_named_library_hash_elf_and_mount_boundary(self):
        m=module('check_switchless_nccl')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); directory=root/'lib'; directory.mkdir(); p=directory/'libnccl.so.2.30.7'
            blob=b'\x7fELF\x02\x01'+b'\0'*12+b'\xb7\x00'+b'test-only'
            p.write_bytes(blob); h=hashlib.sha256(blob).hexdigest()
            self.assertEqual(m.verify_library(p,h),h)
            with self.assertRaises(ValueError):m.verify_library(p,'0'*64)
            p.write_bytes(b'x'*len(blob))
            with self.assertRaises(ValueError):m.verify_library(p,hashlib.sha256(p.read_bytes()).hexdigest())
            p.unlink(); external=root/'outside'; external.write_bytes(blob); p.symlink_to(external)
            with self.assertRaises(ValueError):m.verify_library(p,h)

    def test_projection_and_incomplete_inventory(self):
        m=module('preflight_runtime')
        with patch('sys.argv',['check','--container','new','--overlay','/runtime']),patch.object(m.subprocess,'check_output',side_effect=['abc\ndef\n','{"Name":"/other","Mounts":[]}\n']) as run:
            with self.assertRaisesRegex(RuntimeError,'incomplete'):m.main()
            cmd=run.call_args_list[1].args[0]
            self.assertEqual(cmd[:3],['docker','inspect','--format'])
            self.assertNotIn('Config',cmd[3]);self.assertNotIn('Env',cmd[3]);self.assertIn('.Mounts',cmd[3])

if __name__=='__main__':unittest.main()
