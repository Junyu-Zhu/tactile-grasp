from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r7_eval", HERE / "evaluate_r6_restricted.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
spec7 = importlib.util.spec_from_file_location("r7_formal_eval", HERE / "evaluate_r7.py")
r7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(r7)


class EvaluationTests(unittest.TestCase):
    def test_fixed_rule_values_are_causal(self):
        p = np.array([0.2, 0.8, 0.6])
        np.testing.assert_allclose(m.apply_rule(p, "ema_0p5"), [0.2, 0.5, 0.55])
        np.testing.assert_allclose(m.apply_rule(p, "confirm2"), [0.0, 0.2, 0.6])
        np.testing.assert_allclose(m.apply_rule(p, "ema_0p5_confirm2"), [0.0, 0.2, 0.5])

    def test_confirmation_first_frame_cannot_alarm(self):
        self.assertEqual(m.apply_rule(np.array([1.0]), "confirm2").tolist(), [0.0])

    def test_noncontiguous_restricted_sequence_is_rejected(self):
        rows = [{"episode_id": "e", "t": 1, "p_raw": .1}, {"episode_id": "e", "t": 3, "p_raw": .2}]
        with self.assertRaises(ValueError): m.add_rule_scores(rows)

    def test_trial_false_alarm_threshold_respects_constraint(self):
        rows = []
        for episode, scores in (("a", [.9, .8]), ("b", [.2, .1]), ("c", [.3, .2]), ("d", [.4, .3])):
            for t, score in enumerate(scores): rows.append({"episode_id": episode, "leakage_group": episode, "t": t, "onset": None, "target": 0, "raw": score})
        rows += [{"episode_id": "event", "leakage_group": "event", "t": 0, "onset": 1, "target": 1, "raw": .85}]
        threshold, status = m.select_threshold(rows, "raw", "trial_fa", .25)
        self.assertEqual(status, "available")
        event, _ = m.trial_event_metrics(rows, "raw", threshold)
        self.assertLessEqual(event["trial_false_alarm_rate"], .25)

    def test_event_recall_unavailable_is_explicit(self):
        rows = [{"episode_id": "n", "leakage_group": "n", "t": 0, "onset": None, "target": 0, "raw": .1},
                {"episode_id": "e", "leakage_group": "e", "t": 0, "onset": 1, "target": 1, "raw": .2}]
        threshold, status = m.select_threshold(rows, "raw", "event_recall", 1.1)
        self.assertIsNone(threshold); self.assertEqual(status, "unavailable_calibration_constraint")

    def test_alarm_runs_reset_on_inactive_frame(self):
        self.assertEqual(m.alarm_runs(np.array([1, 2, 3, 4]), np.array([1, 1, 0, 1]))[:3], (2, 3, 2))

    def test_compact_false_alarm_ties_match_full_trial_accounting(self):
        rows = [{"episode_id": "a", "leakage_group": "a", "t": t, "target": 0, "onset": None, "raw": s}
                for t, s in enumerate([.8, .9, .2, .7])]
        rows += [{"episode_id": "b", "leakage_group": "b", "t": t, "target": 0, "onset": None, "raw": s}
                 for t, s in enumerate([.1, .6])]
        for threshold in (.5, .85):
            full, _ = m.trial_event_metrics(rows, "raw", threshold)
            compact = m.compact_false_alarm_stats(rows, "raw", [threshold])[threshold]
            self.assertEqual(compact, (full["false_alarm_starts"], full["false_alarm_duration"]))

    def test_ap_tied_scores_is_finite(self):
        self.assertAlmostEqual(m.average_precision(np.array([0, 1, 1]), np.array([.5, .5, .5])), 2 / 3)

    def test_complete_timeline_rule_precedes_mask(self):
        rows=[]
        for t,(p,eligible,target) in enumerate([(.8,False,None),(.7,True,0),(.6,True,1)]):
            rows.append({"episode_id":"e","t":t,"timeline_contiguous":t>0,"p_H1":p,"eligible_H1":eligible,"common_population":eligible,"target_H1":target})
        r7.add_rules(rows,"p_H1","H1_")
        self.assertAlmostEqual(rows[1]["H1_ema_0p5"],.75)
        self.assertAlmostEqual(rows[1]["H1_confirm2"],.7)

    def test_primary_role_rows_uses_cross_horizon_common_mask(self):
        rows=[{"common_population":False,"eligible_H1":True,"target_H1":0},{"common_population":True,"eligible_H1":True,"target_H1":1}]
        selected=r7.role_rows(rows,1,"primary")
        self.assertEqual([x["metric_mask"] for x in selected],[False,True])

    def test_event_tradeoff_grid_keeps_positive_lead_changes_and_trial_maxima(self):
        rows=[{"episode_id":"e","metric_mask":True,"target":1,"raw":.7},{"episode_id":"e","metric_mask":True,"target":1,"raw":.5},
              {"episode_id":"n","metric_mask":True,"target":0,"raw":.6},{"episode_id":"n","metric_mask":True,"target":0,"raw":.2}]
        values=r7.event_tradeoff_thresholds(rows,"raw","metric_mask")
        self.assertIn(.7,values);self.assertIn(.5,values);self.assertIn(.6,values);self.assertNotIn(.2,values)

    def test_fit_logistic_outputs_finite_probabilities(self):
        x=np.array([[0.],[1.],[2.],[3.]]);y=np.array([0.,0.,1.,1.]);model=r7.fit_logistic(x,y)
        p=r7.predict_logistic(model,x);self.assertTrue(np.isfinite(p).all());self.assertTrue(np.all(np.diff(p)>0))

    def test_calibration_cache_is_content_bound_across_distinct_datasets(self):
        def rows(scores):
            return [{"episode_id":f"e{i}","leakage_group":f"e{i}","t":i,"raw":s,"metric_mask":True,"target":i%2,"onset":i+1 if i%2 else None}
                    for i,s in enumerate(scores)]
        first=rows([.1,.9,.2,.8]);second=rows([.9,.1,.8,.2])
        self.assertNotEqual(r7.calibration_content_key(first,"raw","metric_mask"),r7.calibration_content_key(second,"raw","metric_mask"))
        r7._CAL_CACHE.clear();t1=r7.select_threshold(first,"raw","metric_mask","maxBA",None)[0];t2=r7.select_threshold(second,"raw","metric_mask","maxBA",None)[0]
        self.assertNotEqual(t1,t2)

    def test_ongoing_alarm_overlap_is_not_credited_as_new_event_start(self):
        rows=[]
        for t,target in enumerate([0,0,1,1]):
            rows.append({"episode_id":"e","leakage_group":"e","t":t,"onset":4,"metric_mask":True,"target":target,"raw":.9})
        summary,records=r7.event_metrics(rows,"raw",.5,"metric_mask")
        self.assertEqual(summary["event_recall"],0.)
        self.assertEqual(summary["active_overlap_recall"],1.)
        self.assertTrue(records[0]["ongoing_from_before_positive_window"])
        self.assertIsNone(records[0]["lead_frames"])
        compact=r7.compact_event_curve(rows,"raw","metric_mask",[.5])[0]
        self.assertEqual(compact["event_recall"],0.)

    def test_new_alarm_start_inside_positive_window_has_actual_lead(self):
        rows=[]
        for t,(target,score) in enumerate([(0,.1),(0,.2),(1,.8),(1,.9)]):
            rows.append({"episode_id":"e","leakage_group":"e","t":t,"onset":4,"metric_mask":True,"target":target,"raw":score})
        summary,records=r7.event_metrics(rows,"raw",.5,"metric_mask")
        self.assertEqual(summary["event_recall"],1.)
        self.assertEqual(records[0]["first_alarm_t"],2)
        self.assertEqual(records[0]["lead_frames"],2)

    def test_known_onset_beyond_observable_positive_window_is_right_censored(self):
        rows=[{"episode_id":"e","leakage_group":"e","t":t,"onset":10,"metric_mask":True,"target":0,"raw":.1} for t in range(4)]
        summary,records=r7.event_metrics(rows,"raw",.5,"metric_mask")
        self.assertTrue(records[0]["right_censored_or_no_observed_onset"])
        self.assertEqual(summary["uncensored_events"],0)

    def test_false_alarm_run_mean_weights_runs_not_trials(self):
        rows=[]
        for episode,scores in (("a",[.9,.9,.1,.9]),("b",[.9,.9,.9,.9])):
            for t,score in enumerate(scores):rows.append({"episode_id":episode,"leakage_group":episode,"t":t,"onset":None,"metric_mask":True,"target":0,"raw":score})
        summary,_=r7.event_metrics(rows,"raw",.5,"metric_mask")
        self.assertEqual(summary["false_alarm_runs"],3)
        self.assertAlmostEqual(summary["mean_false_alarm_run_duration"],7/3)
        self.assertEqual(summary["max_false_alarm_duration"],4)

    def test_calibration_cache_warm_result_matches_cold_result(self):
        rows=[{"episode_id":f"e{i}","leakage_group":f"e{i}","t":i,"raw":score,"metric_mask":True,"target":i%2,"onset":i+1 if i%2 else None}
              for i,score in enumerate([.1,.9,.2,.8])]
        r7._CAL_CACHE.clear();cold=r7.select_threshold(rows,"raw","metric_mask","maxBA",None)
        other=[dict(row,raw=1-row["raw"]) for row in rows];r7.select_threshold(other,"raw","metric_mask","maxBA",None)
        warm=r7.select_threshold(rows,"raw","metric_mask","maxBA",None)
        self.assertEqual(cold,warm)

    def test_monotone_platt_preserves_probability_order(self):
        rows=[{"metric_mask":True,"target":target,"raw":score} for target,score in ((0,.1),(0,.2),(1,.7),(1,.9))]
        model=r7.fit_monotone_platt(rows,"raw","metric_mask");self.assertEqual(model["status"],"available")
        transformed=[r7.apply_platt(x,model) for x in (.1,.2,.7,.9)]
        self.assertTrue(np.all(np.diff(transformed)>0))

    def test_vectorized_compact_curve_matches_full_new_start_metrics(self):
        rows=[]
        for episode,onset,scores,targets in (("e",5,[.1,.7,.8,.2,.9],[0,1,1,0,0]),("n",None,[.2,.4,.6,.1,.3],[0,0,0,0,0])):
            for t,(score,target) in enumerate(zip(scores,targets)):rows.append({"episode_id":episode,"leakage_group":episode,"t":t,"onset":onset,"metric_mask":True,"target":target,"raw":score})
        thresholds=[.15,.5,.75,.95];compact=r7.compact_event_curve(rows,"raw","metric_mask",thresholds)
        for threshold,item in zip(thresholds,compact):
            full,_=r7.event_metrics(rows,"raw",threshold,"metric_mask")
            self.assertAlmostEqual(item["trial_false_alarm_rate"],full["trial_false_alarm_rate"])
            self.assertAlmostEqual(item["event_recall"],full["event_recall"])
            self.assertEqual(item["false_alarm_starts"],full["false_alarm_starts"])
            self.assertEqual(item["false_alarm_duration"],full["false_alarm_duration"])
            if np.isfinite(item["mean_lead_frames"]):self.assertAlmostEqual(item["mean_lead_frames"],full["mean_lead_frames"])


if __name__ == "__main__": unittest.main()
