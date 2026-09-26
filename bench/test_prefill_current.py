import base64,json,unittest
from unittest.mock import patch
import prefill_checked as p
import run_prefill as r

def wire():
    events=[{'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]},{'choices':[{'index':0,'delta':{'content':'word'},'finish_reason':'length'}]},{'choices':[],'usage':{'prompt_tokens':32500,'completion_tokens':1,'total_tokens':32501,'prompt_tokens_details':{'cached_tokens':0}}},'[DONE]']
    lines=[]
    for i,event in enumerate(events,1):
        data=event if isinstance(event,str) else json.dumps(event)
        for raw in (('data: '+data+'\n').encode(),b'\n'):lines.append({'elapsed_s':float(i),'base64':base64.b64encode(raw).decode()})
    return lines
class Tests(unittest.TestCase):
 def test_actual_sse_role_excluded(self):
    x=p.validate_events(p.parse_wire_lines(wire()));self.assertEqual(x['first_choices_s'],1);self.assertEqual(x['ttft_s'],2);self.assertEqual(x['prefill_tps'],16250)
 def test_truncated_missingusage(self):
    for lines in (wire()[:-2],wire()[:4]+wire()[6:]):
        with self.assertRaises(ValueError):p.validate_events(p.parse_wire_lines(lines))
 def test_role_only(self):
    e=p.parse_wire_lines(wire());e[1]['data']=json.dumps({'choices':[{'index':0,'delta':{},'finish_reason':'length'}]})
    with self.assertRaises(ValueError):p.validate_events(e)
 def test_cold_usage(self):
    self.assertTrue(r.cold_usage({'usage':{'prompt_tokens_details':{'cached_tokens':0}}}))
    self.assertFalse(r.cold_usage({'usage':{}}))
    for v in (1,False,None):
        with self.assertRaises(ValueError):r.cold_usage({'usage':{'cached_tokens':v}})
 def test_actual_fixed_runner(self):
    calls=[]
    def stream(base,model,prompt,row):
        calls.append(prompt);row.update(p.validate_events(p.parse_wire_lines(wire())));row['prefill_tps']=len(calls)
    out={'records':[]}
    with patch.object(p,'build',return_value='body'),patch.object(p,'stream',side_effect=stream):r.run('base','model',out)
    self.assertEqual(len(calls),11);self.assertEqual(len(set(calls)),11);self.assertEqual([x['median_prefill_tps'] for x in out['summary']],[4,7,10]);self.assertEqual([x['phase'] for x in out['records']],['warmup']*2+['scored']*9);self.assertEqual([x['size'] for x in out['records']],[32768]*2+[16384]*3+[32768]*3+[65536]*3)
if __name__=='__main__':unittest.main()
