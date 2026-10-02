import json,tempfile,unittest
from pathlib import Path
import evaluate as e
class IdentityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name);self.audits={};runs=[]
  for source,groups,schema,summaryschema in [('round9',('V_temporal','F_history','F_delta'),'round9_training_audit_v1','round9_htt_training_summary_v1'),('round10',('F_history_new',),'round10_fusion_training_audit_v1','round10_htt_fusion_summary_v1'),('round11',e.NEW_GROUPS,'round11_fusion_training_audit_v1','round11_htt_fusion_summary_v1')]:
   accepted=[]
   for group in groups:
    for fold in [f'htt_leave_p{i}' for i in range(1,5)]:
     for seed in e.SEEDS:
      sg='F_history' if source=='round10' else group
      q=self.p/f'{source}_{group}_{fold}_{seed}.json';e.js(q,dict(schema=summaryschema,status='complete',smoke=False,group=sg,fold=fold,seed=seed));rec={'path':str(q),'sha256':e.sha(q)};item=dict(group=group,fold=fold,seed=seed,training_summary=rec,checkpoint=rec,predictions={role:rec for role in e.ROLES});accepted.append(item)
      alias='V_original' if source=='round9' and group=='V_temporal' else 'F_history_original' if source=='round10' else group
      if alias in e.GROUPS:runs.append({**item,'group':alias,'source_group':sg,'source_round':source})
   path=self.p/f'{source}_audit.json';e.js(path,dict(schema=schema,status='pass',expected_count=len(accepted),test_role_consumed=False,accepted_runs=accepted));self.audits[source]={'path':str(path),'sha256':e.sha(path)}
  self.m=dict(schema='round11_evaluation_manifest_v1',synthetic=False,runs=runs,training_audits=self.audits,evaluation_source_sha256=e.sha(e.__file__),evaluation_protocol_sha256=e.sha(Path(e.__file__).with_name('PROTOCOL.md')))
 def test_valid(self):self.assertEqual(len(e.validate_manifest(self.m)[1]),60)
 def test_bad_grid(self):
  self.m['runs'][-1]=self.m['runs'][0]
  with self.assertRaisesRegex(ValueError,'grid'):e.validate_manifest(self.m)
 def test_swap_train(self):
  self.m['runs'][-1]['predictions']=dict(self.m['runs'][0]['predictions'])
  with self.assertRaisesRegex(ValueError,'prediction not accepted'):e.validate_manifest(self.m)
 def test_wrong_round(self):
  self.m['runs'][-1]['source_round']='round10'
  with self.assertRaisesRegex(ValueError,'source identity'):e.validate_manifest(self.m)
 def test_wrong_audit(self):
  q=Path(self.audits['round11']['path']);x=json.loads(q.read_text());x['accepted_runs'][-1]=x['accepted_runs'][0];e.js(q,x);self.audits['round11']['sha256']=e.sha(q)
  with self.assertRaisesRegex(ValueError,'audit grid'):e.validate_manifest(self.m)
if __name__=='__main__':unittest.main()
