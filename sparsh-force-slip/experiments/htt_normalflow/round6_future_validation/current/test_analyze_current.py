import json,tempfile,unittest
from pathlib import Path
import numpy as np
import analyze_current as a

def rows(scores,labels,episode='e'):
 return [{'episode':episode,'t':10+i,'stage':2 if y else 0,'score':float(s)} for i,(s,y) in enumerate(zip(scores,labels))]

class TestCurrentAnalysis(unittest.TestCase):
 def test_curve_has_endpoints_and_counts(self):
  c=a.curve(rows([.9,.8,.2,.1],[1,0,1,0]))
  self.assertEqual(c[0]['gross_recall'],0);self.assertEqual(c[0]['static_fpr'],0)
  self.assertEqual(c[-1]['gross_recall'],1);self.assertEqual(c[-1]['static_fpr'],1)
 def test_curve_groups_tied_scores_atomically(self):
  c=a.curve(rows([.8,.8,.2,.2],[1,0,1,0]));self.assertEqual(len(c),3)
  self.assertEqual((c[1]['tp'],c[1]['fp']),(1,1));self.assertEqual((c[2]['tp'],c[2]['fp']),(2,2))
 def test_recall_selector_minimizes_fpr(self):
  r=rows([.9,.8,.7,.1],[1,0,1,0]);p=a.choose_point(r,'recall',1.0)
  self.assertEqual(p['gross_recall'],1);self.assertEqual(p['static_fpr'],.5);self.assertAlmostEqual(p['threshold'],.7)
 def test_fpr_selector_maximizes_recall(self):
  r=rows([.9,.8,.7,.1],[1,0,1,0]);p=a.choose_point(r,'fpr',0.0)
  self.assertEqual(p['static_fpr'],0);self.assertEqual(p['gross_recall'],.5)
 def test_threshold_never_alarm(self):
  p=a.evaluate_threshold(rows([.3,.7],[0,1]),'inf');self.assertTrue(p['never_alarm']);self.assertEqual(p['gross_recall'],0)
 def test_single_class_rejected(self):
  with self.assertRaises(ValueError):a.curve(rows([.1,.2],[1,1]))
 def test_sequence_separates_left_censor(self):
  r=rows([.9,.9,.1,.1,.9,.9],[1,1,0,0,1,1]);m=a.sequence_metrics(r,.5)
  self.assertEqual(m['gross_segments_including_left_censored'],2);self.assertEqual(m['left_censored_gross_segments'],1)
  self.assertEqual(m['uncensored_gross_events'],1);self.assertEqual(m['uncensored_gross_event_recall'],1);self.assertEqual(m['uncensored_mean_delay_frames'],0)
 def test_rank_ties(self):
  np.testing.assert_allclose(a.ranks([3,1,1]),[3,1.5,1.5])
 def test_weighted_static_mean_skips_empty_trials(self):
  value=a.weighted_static_mean([{'static_frames':0,'mean_score':None},{'static_frames':2,'mean_score':.25},{'static_frames':1,'mean_score':1.0}]);self.assertAlmostEqual(value,.5)
 def test_episode_metadata_exposes_probe(self):
  self.assertEqual(a.episode_metadata('htt/p3_sliding/0_press_16'),{'task':'htt','probe':'p3_sliding'})
 def test_expected_grid(self):self.assertEqual(len(a.expected()),48)
 def test_discover_rejects_missing(self):
  with self.assertRaises(ValueError):a.discover({'jobs':[]})
 def test_stable_seed(self):self.assertEqual(a.stable_seed('x',1),a.stable_seed('x',1))
 def test_protocol_constants(self):
  self.assertEqual(a.RECALL_LEVELS,(.8,.9,.95));self.assertEqual(a.FPR_LEVELS,(.01,.05,.1));self.assertEqual(a.BOOTSTRAPS,200)

if __name__=='__main__':unittest.main()
