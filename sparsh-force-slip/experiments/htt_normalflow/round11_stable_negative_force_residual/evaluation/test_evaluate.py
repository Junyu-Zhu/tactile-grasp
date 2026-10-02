import tempfile,unittest,csv
from pathlib import Path
import numpy as np
import evaluate as e

def rows(stages,scores,ts=None):return [dict(episode='a',t=t,stage=y,score=p,group='g') for t,y,p in zip(ts or range(len(stages)),stages,scores)]
class Tests(unittest.TestCase):
 def test_ties(self):
  r=rows([0,2],[.5,.5]);p=e.choose(r);self.assertGreater(p['FPR0.01'],1);self.assertEqual(p['recall0.80'],.5)
 def test_rank_ties(self):
  x=e.rank_metrics(rows([0,2],[.5,.5]));self.assertAlmostEqual(x['AP'],.5);self.assertAlmostEqual(x['pAUC'],.05)
 def test_perfect(self):
  x=e.rank_metrics(rows([0,2],[.1,.9]));self.assertAlmostEqual(x['AP'],1);self.assertAlmostEqual(x['pAUC'],1)
 def test_left_censor(self):
  t=e.trial_metrics(rows([2,2,0,2,2],[.9]*5),.5,1)[0];self.assertEqual(t['left_censored'],1);self.assertEqual(t['events'],1);self.assertEqual(t['hits'],1)
 def test_gap_resets(self):
  t=e.trial_metrics(rows([0,0],[.9,.9],[13,15]),.5,2)[0];self.assertEqual(t['fp'],0)
 def test_confirm_delay(self):
  t=e.trial_metrics(rows([0,2,2],[.1,.9,.9]),.5,2)[0];self.assertEqual(t['delay_sum'],1)
 def test_incipient_excluded(self):
  a=e.aggregate(e.trial_metrics(rows([0,1,2],[.1,.9,.9]),.5,1));self.assertEqual(a['tn'],1);self.assertEqual(a['tp'],1);self.assertEqual(a['incipient_frames'],1)
 def test_weighted_rank(self):
  r=rows([0,2],[.2,.3])+[dict(episode='b',t=0,stage=0,score=.9,group='b')];x=e.rank_metrics(r,{'g':2,'b':0});self.assertEqual(x['AP'],1)
 def test_role_and_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('episode_id,t,stage,leakage_group,role\na,13,0,g,validation\n');q=Path(d)/'p';q.write_text('episode_id,t,stage,p_slip,role\na,13,0,0.2,validation\n');ep={'path':str(p),'sha256':e.sha(p)};pr={'path':str(q),'sha256':e.sha(q)};self.assertEqual(len(e.load_rows(pr,ep,'validation')),1)
   with self.assertRaises(ValueError):e.load_rows(pr,ep,'calibration')
 def test_historical_frame_alias(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('episode_id,t,stage,leakage_group,role\na,13,0,g,validation\n');q=Path(d)/'p';q.write_text('episode,frame,stage,probability\na,13,0,0.2\n');ep={'path':str(p),'sha256':e.sha(p)};pr={'path':str(q),'sha256':e.sha(q)};self.assertEqual(len(e.load_rows(pr,ep,'validation',True)),1)
 def test_never_alarm(self):
  x=e.aggregate(e.trial_metrics(rows([0,2],[1.,1.]),np.nextafter(1.,np.inf),1));self.assertTrue(x['observed_no_alarm']);self.assertEqual(x['gross_recall'],0)
if __name__=='__main__':unittest.main()
