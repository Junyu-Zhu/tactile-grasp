import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
import evaluate as e
class Checks(unittest.TestCase):
 def setupfiles(self,d,role=True,fold='htt_leave_p1',extra=False):
  p=Path(d)/'end.csv';q=Path(d)/'pred.csv';e.csvout(p,[dict(episode_id='a',t=13,stage=0,leakage_group='a',role='validation',fold=fold)])
  r=dict(episode_id='a',t=13,stage=0,p_slip=.2)
  if role:r['role']='validation'
  e.csvout(q,[r]+([dict(r,t=14)] if extra else []));return {'path':str(q),'sha256':e.sha(q)},{'path':str(p),'sha256':e.sha(p)}
 def test_missing_new_role_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,x=self.setupfiles(d,role=False)
   with self.assertRaises(ValueError):e.load_rows(p,x,'validation')
 def test_endpoint_fold_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,x=self.setupfiles(d)
   with self.assertRaises(ValueError):e.load_rows(p,x,'validation',fold='htt_leave_p2')
 def test_superset_only_history(self):
  with tempfile.TemporaryDirectory() as d:
   p,x=self.setupfiles(d,extra=True)
   with self.assertRaises(ValueError):e.load_rows(p,x,'validation')
   self.assertEqual(len(e.load_rows(p,x,'validation',historical=True)),1)
 def test_incipient_can_confirm_but_no_primary_count(self):
  r=[dict(episode='a',t=i+13,stage=y,score=.9,group='g') for i,y in enumerate([1,2])]
  a=e.aggregate(e.trial_metrics(r,.5,2));self.assertEqual(a['tp'],1);self.assertEqual(a['fp'],0);self.assertEqual(a['event_recall'],1);self.assertEqual(a['mean_delay'],0)
if __name__=='__main__':unittest.main()
