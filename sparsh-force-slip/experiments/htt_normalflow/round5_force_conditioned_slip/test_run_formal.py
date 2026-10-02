import hashlib,json,tempfile,unittest
from pathlib import Path
from run_formal import accepted,command_identity,launch_argv

class AcceptanceTests(unittest.TestCase):
    def test_missing_and_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.json';j={'acceptance_path':str(p)}
            self.assertFalse(accepted(j));p.write_text('{"status":"running"}')
            self.assertFalse(accepted(j))
    def test_smoke_and_nonformal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.json';j={'acceptance_path':str(p)}
            for fields in ({'smoke':True},{'formal':False}):
                p.write_text(json.dumps({'status':'complete',**fields}))
                with self.assertRaises(ValueError):accepted(j)
    def test_artifact_missing_or_tampered(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.json';a=Path(d)/'best.pth';a.write_bytes(b'proof')
            p.write_text(json.dumps({'status':'complete','smoke':False,'artifacts':{'best':{'path':str(a),'sha256':hashlib.sha256(b'proof').hexdigest()}}}))
            j={'acceptance_path':str(p)};self.assertTrue(accepted(j))
            a.write_bytes(b'changed');self.assertFalse(accepted(j))
            a.unlink();self.assertFalse(accepted(j))
    def test_receipt_binds_exact_argv_and_file_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);summary=d/'summary.json';script=d/'train.py';cache=d/'cache.pt';out=d/'run'
            script.write_text('pass\n');cache.write_bytes(b'cache-v1')
            summary.write_text(json.dumps({'status':'complete','formal':True,'smoke':False}))
            job={'id':'future','acceptance_path':str(summary),'argv':['python',str(script),'train-source','--train-cache',str(cache),'--output',str(out)]}
            receipt=summary.with_name(summary.name+'.scheduler_receipt.json')
            self.assertFalse(accepted(job))
            receipt.write_text(json.dumps(command_identity(job)))
            self.assertTrue(accepted(job))
            cache.write_bytes(b'cache-v2')
            with self.assertRaisesRegex(ValueError,'drift'):accepted(job)
    def test_absolute_path_key_provenance_is_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);summary=d/'metrics.json';csv=d/'pred.csv';csv.write_text('x\n')
            summary.write_text(json.dumps({'status':'complete','formal':True,'provenance':{str(csv):hashlib.sha256(b'x\n').hexdigest()}}))
            self.assertTrue(accepted({'acceptance_path':str(summary)}))
            csv.write_text('changed\n')
            self.assertFalse(accepted({'acceptance_path':str(summary)}))
    def test_future_resume_is_added_only_after_latest_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';run.mkdir()
            job={'argv':['python','future_pipeline.py','train-htt','--output',str(run)]}
            self.assertEqual(launch_argv(job),job['argv'])
            (run/'latest.pth').write_bytes(b'committed')
            self.assertEqual(launch_argv(job),[*job['argv'],'--resume'])
            already={"argv":[*job['argv'],'--resume']}
            self.assertEqual(launch_argv(already),already['argv'])
            unrelated={'argv':['python','train_force.py','--output',str(run)]}
            self.assertEqual(launch_argv(unrelated),unrelated['argv'])
if __name__=='__main__':unittest.main()
