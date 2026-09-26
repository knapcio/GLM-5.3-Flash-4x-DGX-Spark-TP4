"""API-contract tests without a server; exercise real polling/validation."""
import collections
import copy
import unittest
import final_sparkdash as f


def job(c=4):
    return dict(benchId='new-id',sparkId='spark-01',status='completed',error=None,
                startedAt=1000,completedAt=2000,
                config=dict(port=8093,promptType='prose',concurrencies=[c],maxTokens=256,modelId='GLM-5.3-Flash-FP8'),
                progress=dict(completedLevels=1,totalLevels=1),
                results=[dict(concurrency=c,streamsOk=c,streamsFailed=0,error=None,
                              meanDecodeTps=40.,aggregateDecodeTps=150.,totalCompletionTokens=c*256,totalDecodeTokens=c*255,
                              streams=[dict(index=i,error=None,reasoningChunks=0,completionTokens=256,
                                            decodeTokens=255,decodeTps=40.) for i in range(c)])])


def posted(j):
    p=copy.deepcopy(j);p.update(status='running',completedAt=None,results=[])
    p['progress']['completedLevels']=0
    return p


class Contracts(unittest.TestCase):
    def test_matrix_counts_and_discarded_warmups(self):
        cells=f.cells();self.assertEqual(cells[:2],[('warmup','prose',1)]*2)
        counts=collections.Counter((t,c) for phase,t,c in cells if phase!='warmup')
        self.assertEqual(counts[('prose',1)],5);self.assertEqual(counts[('prose',4)],3)
        self.assertEqual(len(cells),20);self.assertEqual(len(counts),12)
        self.assertTrue(all(n==1 for pair,n in counts.items() if pair not in [('prose',1),('prose',4)]))

    def test_valid_job_and_failed_stream_rejection(self):
        self.assertEqual(f.validate_job(job(),'new-id','prose',4),[])
        for key,value in [('streamsFailed',1),('streamsOk',3),('totalCompletionTokens',1023),('totalDecodeTokens',1024),
                          ('aggregateDecodeTps',float('nan')),('error','partial failure')]:
            j=job();j['results'][0][key]=value
            self.assertTrue(f.validate_job(j,'new-id','prose',4),(key,value))
        for key,value in [('completionTokens',255),('completionTokens',None),('reasoningChunks',1),
                          ('index',3),('error','early close'),('decodeTokens',False),('decodeTokens',1),('decodeTokens',256)]:
            j=job();j['results'][0]['streams'][0][key]=value
            self.assertTrue(f.validate_job(j,'new-id','prose',4),(key,value))

    def test_identity_config_and_completion_are_mandatory(self):
        for key,value in [('benchId','old-id'),('sparkId','spark-02'),('status','cancelled'),
                          ('completedAt',None),('error','failed')]:
            j=job();j[key]=value;self.assertTrue(f.validate_job(j,'new-id','prose',4))
        for key,value in [('port',8094),('maxTokens',128),('promptType','code'),('concurrencies',[1]),('modelId','other')]:
            j=job();j['config'][key]=value;self.assertTrue(f.validate_job(j,'new-id','prose',4))
        j=job();j['progress']['completedLevels']=0;self.assertTrue(f.validate_job(j,'new-id','prose',4))

    def test_polls_exact_id_and_keeps_full_result(self):
        j=job();calls=[];record={}
        def http(url,body=None,timeout=None):
            calls.append(url)
            if body is not None:self.assertEqual(body['modelId'],'GLM-5.3-Flash-FP8')
            return (202,posted(j)) if body is not None else (200,copy.deepcopy(j))
        result=f.collect_job('http://fixture','prose',4,10,record,lambda:None,http=http)
        self.assertEqual(calls,['http://fixture/bench','http://fixture/bench/new-id'])
        self.assertEqual(record['job'],result)
        self.assertIn('streams',record['job']['results'][0])

    def test_stale_and_reused_ids_fail_without_last_endpoint(self):
        j=job()
        def http(url,body=None,timeout=None):
            got=copy.deepcopy(j)
            if body is None:got['benchId']='stale'
            return (202,posted(j)) if body is not None else (200,got)
        with self.assertRaisesRegex(RuntimeError,'identity'):
            f.collect_job('http://fixture','prose',4,10,{},lambda:None,http=http)
        with self.assertRaisesRegex(RuntimeError,'reused'):
            f.collect_job('http://fixture','prose',4,10,{},lambda:None,seen_ids={'new-id'},http=http)

    def test_deadline_is_bounded(self):
        clock=iter([0,0,2,3]);j=job();j['status']='running'
        def http(url,body=None,timeout=None):return (202,posted(j)) if body is not None else (200,j)
        with self.assertRaises(TimeoutError):
            f.collect_job('http://fixture','prose',4,1,{},lambda:None,http=http,
                          clock=lambda:next(clock),sleep=lambda _:None)

    def test_post_rejects_precompleted_wrong_config_or_wrong_spark(self):
        for kind in ('completed','config','spark'):
            p=posted(job())
            if kind=='completed':p=job()
            elif kind=='config':p['config']['modelId']='other'
            else:p['sparkId']='spark-02'
            with self.assertRaisesRegex(RuntimeError,'fresh running'):
                f.collect_job('http://fixture','prose',4,10,{},lambda:None,
                              http=lambda *a,**kw:(202,p))

    def test_summary_excludes_warmup_and_preserves_metric_fields(self):
        records=[dict(phase='warmup',kind='prose',concurrency=4,valid=True,job=job()),
                 dict(phase='matrix',kind='prose',concurrency=4,valid=True,job=job())]
        s=f.summarize(records)['prose:c4'];self.assertEqual(s['n'],1)
        self.assertEqual(s['aggregateDecodeTps_median'],150.)
        self.assertEqual(s['meanDecodeTps_median'],40.)


if __name__=='__main__':unittest.main()
