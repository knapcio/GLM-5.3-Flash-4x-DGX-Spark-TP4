import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('safe_stop',ROOT/'scripts/stop_preserving.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
class RuntimeTests(unittest.TestCase):
 def test_exact_aux_matching(self):
  r=Path('/x');c='model-r0'
  self.assertTrue(s.matches(['python3','/x/scripts/prewarm.py','/models',c,'3','4'],r,c,None,42))
  self.assertFalse(s.matches(['python3','/other/scripts/prewarm.py','/models',c],r,c,None,42))
  self.assertFalse(s.matches(['python3','/x/scripts/prewarm.py','/models','other'],r,c,None,42))
  self.assertTrue(s.matches(['python3','/x/scripts/boot_warm.py'],r,c,42,42))
  self.assertFalse(s.matches(['python3','/x/scripts/boot_warm.py'],r,c,41,42))
 def test_unrelated_permission_and_pinned_identity(self):
  with tempfile.TemporaryDirectory() as td:
   r=Path(td);proc=r/'proc';proc.mkdir()
   for pid in (40,41):(proc/str(pid)).mkdir()
   (proc/'40/cmdline').write_bytes(b'foreign\0')
   (proc/'41/cmdline').write_bytes(('python3\0'+str(r/'scripts/prewarm.py')+'\0/models\0model-r0\0').encode())
   original=Path.read_bytes
   def read(p):
    if p==proc/'40/cmdline':raise PermissionError('unrelated')
    return original(p)
   with patch.object(Path,'read_bytes',read),patch.object(s.os,'pidfd_open',return_value=99,create=True) as op,patch.object(s.signal,'pidfd_send_signal',create=True) as send,patch.object(s.os,'close'):
    s.stop_aux(r,'model-r0',proc);op.assert_called_once_with(41);send.assert_called_once_with(99,s.signal.SIGTERM)
   with patch.object(Path,'read_bytes',read),patch.object(s.os,'pidfd_open',side_effect=PermissionError('matched'),create=True):
    with self.assertRaises(PermissionError):s.stop_aux(r,'model-r0',proc)
 def test_stop_order_and_no_remove(self):
  events=[]
  with patch('sys.argv',['stop','--root','/x','--container','model-r0']),patch.object(s,'stop_aux',side_effect=lambda *x:events.append('aux')),patch.object(s.subprocess,'run',side_effect=lambda argv,**kw:events.append(argv)):
   s.main()
  self.assertEqual(events,['aux',['docker','stop','--time','30','model-r0']])
 def test_actual_launcher_refuses_existing_before_sync(self):
  with tempfile.TemporaryDirectory() as td:
   r=Path(td);b=r/'bin';b.mkdir();log=r/'log'
   for name,body in {'ssh':'exec bash -c "${@: -1}"','docker':'''echo "$*" >> "$TEST_LOG"; if [[ $1 == ps ]]; then echo abc; else echo '[{"Name":"/existing-r0","Mounts":[]}]'; fi''' ,'rsync':'echo RSYNC >> "$TEST_LOG"; exit 99'}.items():
    p=b/name;p.write_text('#!/usr/bin/env bash\n'+body+'\n');p.chmod(0o755)
   env=r/'config';env.write_text('HOSTS="a b c d"\nIPS="1 2 3 4"\nCTN=existing\nOVERLAY_REMOTE=/unused\n')
   p=subprocess.run(['bash',str(ROOT/'start.sh'),'serve'],env={**os.environ,'PATH':str(b)+':'+os.environ['PATH'],'ENV_FILE':str(env),'TEST_LOG':str(log)},capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertNotIn('RSYNC',log.read_text());self.assertIn('inspect abc',log.read_text())
 def test_actual_dry_run_no_external_calls(self):
  with tempfile.TemporaryDirectory() as td:
   r=Path(td);b=r/'bin';b.mkdir();log=r/'calls'
   for name in ('ssh','docker','rsync'):
    p=b/name;p.write_text('#!/bin/sh\necho CALLED >> "$TEST_LOG"\nexit 99\n');p.chmod(0o755)
   env=r/'config';env.write_text('source .env.example\n')
   p=subprocess.run(['bash',str(ROOT/'start.sh'),'serve'],env={**os.environ,'PATH':str(b)+':'+os.environ['PATH'],'ENV_FILE':str(env),'TEST_LOG':str(log),'DRY':'1'},capture_output=True,text=True)
   self.assertEqual(p.returncode,0,p.stderr);self.assertFalse(log.exists());self.assertIn('[dry-run] rsync',p.stdout)
 def test_profile_has_only_accepted_flags(self):
  p=subprocess.run(['bash','-c','IB_HCA=a,b; source profiles/current.env; printf "%s\\n" "$EXTRA_ENV"'],cwd=ROOT,capture_output=True,text=True,check=True)
  env=dict(x.split('=',1) for x in p.stdout.split())
  self.assertEqual(env['B12X_ROCE_HCA'],'a,b');self.assertEqual(env['GLM_L2_PREFETCH'],'1');self.assertEqual(env['GLM_LV_MODE'],'batch-uniform')
  for k in ('GLM_GDN_METADATA_FAST','GLM_ROUTER_DEDUP','GLM_MHC_FUSED','GLM_AB_VARIANTS','VLLM_SERVER_DEV_MODE','GLM_DS_CPU_PIN'):self.assertNotIn(k,env)
 def test_syntax_no_model_import(self):
  for p in (ROOT/'overlay').glob('*.py'):ast.parse(p.read_text())
  for name in ('start.sh','scripts/build_lossless8.sh'):subprocess.run(['bash','-n',str(ROOT/name)],check=True)

class PreflightTests(unittest.TestCase):
 def module(self):
  spec=importlib.util.spec_from_file_location('preflight',ROOT/'scripts/preflight_runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_preserved_mount_and_names(self):
  m=self.module()
  for c in ({'Name':'/new','Mounts':[]},{'Name':'/old','Mounts':[{'Type':'bind','Source':'/runtime'}]},{'Name':'/old','Mounts':[{'Type':'bind','Source':'/runtime/overlay'}]}):
   with self.assertRaises(RuntimeError):m.verify([c],'new','/runtime')
  m.verify([{'Name':'/old','Mounts':[{'Type':'bind','Source':'/other'}]}],'new','/runtime')
 def test_docker_failure_not_absence(self):
  m=self.module()
  with patch('sys.argv',['preflight','--container','new','--overlay','/runtime']),patch.object(m.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,['docker'])):
   with self.assertRaises(subprocess.CalledProcessError):m.main()
 def test_empty_inventory(self):
  m=self.module()
  with patch('sys.argv',['preflight','--container','new','--overlay','/runtime']),patch.object(m.subprocess,'check_output',return_value='') as run:
   m.main();run.assert_called_once_with(['docker','ps','-aq'],text=True)

class WeightWrapperTests(unittest.TestCase):
 def test_three_calls_and_assemble_validation(self):
  with tempfile.TemporaryDirectory() as td:
   r=Path(td);base=r/'base';base.mkdir();b=r/'bin';b.mkdir();log=r/'calls';fake=b/'python3'
   fake.write_text('#!/bin/sh\nprintf "%s\\n" "$2" >> "$TEST_LOG"\n');fake.chmod(0o755)
   p=subprocess.run(['bash',str(ROOT/'scripts/build_lossless8.sh'),str(base),str(r/'out')],env={**os.environ,'PATH':str(b)+':'+os.environ['PATH'],'TEST_LOG':str(log)},capture_output=True,text=True)
   self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(log.read_text().splitlines(),['plan','quant','assemble'])
  tree=ast.parse((ROOT/'scripts/glm_quant_mix.py').read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cmd_assemble')
  text=ast.get_source_segment((ROOT/'scripts/glm_quant_mix.py').read_text(),fn)
  self.assertIn('verify',text);self.assertIn('zero_copy',text)

if __name__=='__main__':unittest.main()
