from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("round6_root_evaluation", ROOT / "evaluate_training.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class RootEvaluationTests(unittest.TestCase):
    def test_thresholds_match_exhaustive_exact_tie_rules(self):
        rng=np.random.default_rng(20260915)
        for _ in range(50):
            y=np.r_[np.zeros(31,dtype=int),np.ones(17,dtype=int)];rng.shuffle(y)
            p=rng.choice([.05,.2,.2,.5,.75,.9],size=len(y))
            got=m.thresholds(y,p);candidates=np.r_[np.nextafter(1.,np.inf),np.unique(p)]
            rows=[(float(t),m.metrics(y,p,t)) for t in candidates]
            expected=max(rows,key=lambda row:(row[1]["BA"],-row[1]["FPR"],row[0]))[0]
            self.assertEqual(got["calibration_maxBA"],expected)
            for limit in (.01,.05,.10):
                valid=[row for row in rows if row[1]["FPR"]<=limit+1e-15]
                expected=max(valid,key=lambda row:(row[1]["recall"],-row[1]["FPR"],row[0]))[0]
                self.assertEqual(got[f"calibration_FPR_{limit:.2f}"],expected)

    def test_weighted_ap_matches_sklearn(self):
        y=np.array([0,1,0,1,1]);p=np.array([.4,.9,.4,.6,.6]);w=np.array([2,1,3,4,2],dtype=float)
        got=m.metrics(y,p,.5,w)
        self.assertAlmostEqual(got["AP"],average_precision_score(y,p,sample_weight=w),places=14)

    def test_never_alarm_sentinel_is_global(self):
        y=np.array([0,0,1,1]);p=np.ones(4)
        got=m.thresholds(y,p)
        self.assertGreater(got["calibration_FPR_0.01"],1.0)
        self.assertTrue(m.metrics(y,p,got["calibration_FPR_0.01"])["never_alarm"])


if __name__=="__main__": unittest.main(verbosity=2)
