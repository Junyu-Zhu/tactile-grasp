import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace

import summarize_current as s


class InventoryTests(unittest.TestCase):
    def make_job(self, model, fold, seed):
        argv = ["python", "evaluate_slip.py", "--model-id", model, "--fold", fold,
                "--seed", str(seed), "--output", "/tmp/out"]
        if model == "mae-r3-b":
            argv.append("--historical-baseline")
        return {"id": f"{model}-{fold}-{seed}", "kind": "evaluation", "argv": argv,
                "depends_on": [], "acceptance_path": f"/missing/{model}-{fold}-{seed}.json"}

    def test_exact_48_identity_contract(self):
        jobs = [self.make_job(model, fold, seed) for model in s.MODELS for fold in s.FOLDS for seed in s.SEEDS]
        found, errors = s.audit_inventory({"jobs": jobs})
        self.assertEqual(errors, [])
        self.assertEqual(set(found), s.expected_identities())

    def test_historical_flag_and_missing_identity_rejected(self):
        jobs = [self.make_job(model, fold, seed) for model in s.MODELS for fold in s.FOLDS for seed in s.SEEDS]
        jobs[0]["argv"].remove("--historical-baseline")
        jobs.pop()
        _, errors = s.audit_inventory({"jobs": jobs})
        self.assertTrue(any("historical identity" in error for error in errors))
        self.assertTrue(any("missing identities" in error for error in errors))

    def test_incomplete_run_writes_pending_only(self):
        jobs = [self.make_job(model, fold, seed) for model in s.MODELS for fold in s.FOLDS for seed in s.SEEDS]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"jobs": jobs}))
            output = root / "output"
            result = s.run(SimpleNamespace(inventory=inventory, output=output, require_complete=False))
            self.assertEqual(result["status"], "pending")
            self.assertEqual(len(result["pending"]), 48)
            self.assertTrue((output / "readiness.json").is_file())
            self.assertFalse((output / "summary.json").exists())


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def frames(scores, episode):
        stages = [0, 0, 2, 2]
        return [{"episode": episode, "t": index + 10, "stage": stage, "score": score}
                for index, (stage, score) in enumerate(zip(stages, scores))]

    @staticmethod
    def trial(episode, group, fp, tp, alarm_frames, starts, detected, delay):
        return {
            "episode": episode, "bootstrap_group": group, "frames": "4",
            "static_frames": "2", "gross_frames": "2", "incipient_frames": "0",
            "tn": str(2 - fp), "fp": str(fp), "fn": str(2 - tp), "tp": str(tp),
            "false_alarm_starts": str(starts), "static_frames_alarming": str(alarm_frames),
            "gross_events": "1", "gross_events_detected": str(detected),
            "left_censored_gross_segments": 0, "uncensored_gross_events": 1,
            "uncensored_gross_events_detected": detected,
            "uncensored_detection_delay_sum_frames": 0 if delay is None else delay,
            "mean_detection_delay_frames": "" if delay is None else str(delay),
        }

    def test_paired_pauc_bootstrap_is_complete_group_and_deterministic(self):
        baseline = self.frames([.2, .3, .7, .8], "a") + self.frames([.2, .3, .7, .8], "b")
        candidate = self.frames([.1, .2, .8, .9], "a") + self.frames([.1, .2, .8, .9], "b")
        first, draws1 = s.paired_bootstrap_pauc(candidate, baseline, {"a": "ga", "b": "gb"}, 4, 20)
        second, draws2 = s.paired_bootstrap_pauc(candidate, baseline, {"a": "ga", "b": "gb"}, 4, 20)
        self.assertEqual(first, second)
        self.assertEqual(draws1, draws2)
        self.assertEqual(first["bootstrap_groups"], 2)
        self.assertEqual(first["replicates_requested"], 20)

    def test_fold_interval_reuses_one_group_draw_across_all_seeds(self):
        fold = "htt_leave_p1"
        runs = {}
        for index, seed in enumerate(s.SEEDS):
            baseline = self.frames([.2, .3, .7, .8], "a") + self.frames([.2, .3, .7, .8], "b")
            candidate = self.frames([.1, .2, .8, .9], "a") + self.frames([.1, .2, .8, .9], "b")
            runs[("V", fold, seed)] = {"prediction_rows": baseline}
            runs[("F-adapt", fold, seed)] = {"prediction_rows": candidate}
        first, draws1 = s.shared_fold_pauc_interval(
            runs, "F-adapt", fold, {"a": "ga", "b": "gb"}, repetitions=20
        )
        second, draws2 = s.shared_fold_pauc_interval(
            runs, "F-adapt", fold, {"a": "ga", "b": "gb"}, repetitions=20
        )
        self.assertEqual(first, second)
        self.assertEqual(draws1, draws2)
        self.assertTrue(first["shared_group_draw_across_three_seeds"])
        self.assertEqual(first["replicates_requested"], 20)

    def test_trial_pair_population_mismatch_rejected(self):
        left = [self.trial("a", "ga", 0, 2, 0, 0, 1, 0)]
        right = [self.trial("a", "ga", 0, 2, 0, 0, 1, 0)]
        right[0]["gross_frames"] = "1"
        with self.assertRaisesRegex(ValueError, "population mismatch"):
            s.paired_bootstrap_trials(left, right, "gross_recall", 1, 10)

    def test_trial_pair_reports_recall_and_false_alarm_differences(self):
        candidate = [self.trial("a", "ga", 0, 2, 0, 0, 1, 0), self.trial("b", "gb", 0, 2, 0, 0, 1, 0)]
        baseline = [self.trial("a", "ga", 1, 1, 1, 1, 1, 1), self.trial("b", "gb", 1, 1, 1, 1, 1, 1)]
        recall, _ = s.paired_bootstrap_trials(candidate, baseline, "gross_recall", 3, 20)
        fpr, _ = s.paired_bootstrap_trials(candidate, baseline, "static_fpr", 3, 20)
        self.assertAlmostEqual(recall["point_difference"], .5)
        self.assertAlmostEqual(fpr["point_difference"], -.5)

    def test_left_censored_gross_segment_is_excluded_from_event_metrics(self):
        rows = [
            {"stage": 2, "score": .9}, {"stage": 2, "score": .9},
            {"stage": 0, "score": .1}, {"stage": 2, "score": .1},
            {"stage": 2, "score": .9},
        ]
        result = s.uncensored_event_statistics(rows, .5, 1, 1.)
        self.assertEqual(result["gross_segments_including_left_censored"], 2)
        self.assertEqual(result["left_censored_gross_segments"], 1)
        self.assertEqual(result["uncensored_gross_events"], 1)
        self.assertEqual(result["uncensored_gross_events_detected"], 1)
        self.assertEqual(result["_uncensored_delays"], [1])


class RenderSmokeTests(unittest.TestCase):
    def test_tables_report_and_figures_render_from_complete_fixture(self):
        rows = []
        for model in s.MODELS:
            for fold in s.FOLDS:
                for seed in s.SEEDS:
                    common = {
                        "model": model, "fold": fold, "seed": seed,
                        "partial_tpr_auc_0_0p1": .7, "balanced_accuracy": .8, "macro_f1": .8,
                        "average_precision": .8, "positive_prevalence": .2, "static_fpr": .04,
                        "gross_recall": .7, "gross_event_recall": .7,
                        "gross_segment_alarm_coverage_including_left_censored": .7,
                        "left_censored_gross_segments": 1, "uncensored_gross_events": 2,
                        "uncensored_gross_event_recall": .6,
                        "false_alarm_starts_per_trial": .2, "static_alarming_fraction": .03,
                        "mean_detection_delay_frames": 1., "median_detection_delay_frames": 1.,
                        "uncensored_mean_detection_delay_frames": 1.5,
                        "uncensored_median_detection_delay_frames": 1.5,
                        "threshold": .5, "confirmation_k": 2, "release_ratio": .8,
                        "never_alarm": False, "observed_no_alarm": False,
                        "observed_no_primary_alarm": False, "metrics_path": "/fixture",
                    }
                    rows.append({**common, "operating_point": "fixed_0.5", "mode": "raw"})
                    for operation in s.BUDGET_OPS:
                        rows.append({**common, "operating_point": operation, "mode": "sequential"})
        directions = [{
            "candidate": model, "baseline": "V", "fold": fold,
            "three_seed_mean_pauc_difference": .01, "ci_lower": -.01, "ci_upper": .03,
            "positive_direction": True, "positive_seed_differences": 2, "seeds": 3,
            "bootstrap_replicates_valid": 200, "shared_group_draw_across_three_seeds": True,
        } for model in s.FORCE_MODELS for fold in s.FOLDS]
        # A percentile interval may exclude the separately computed point estimate.
        # Rendering must preserve both values rather than creating an invalid negative yerr.
        directions[0]["three_seed_mean_pauc_difference"] = .05
        cases, curves = [], {}
        for model in s.MODELS:
            identity = (model, s.FOLDS[0], s.SEEDS[0])
            episode = model + "_episode"
            case = {"model": model, "fold": identity[1], "seed": identity[2],
                    "failure_type": "false_positive", "episode": episode,
                    "static_alarming_fraction": .5, "false_alarm_starts": 1,
                    "threshold": .5, "confirmation_k": 1, "release_ratio": 1.}
            cases.append(case)
            frame_rows = [{"episode": episode, "t": 10 + index, "stage": stage, "score": score}
                          for index, (stage, score) in enumerate([(0, .2), (0, .8), (2, .9)])]
            curves[(identity, "false_positive", episode)] = (frame_rows, np.array([False, True, True]))
        summary = s.descriptive_summary(rows)
        decisions = {model: {"positive_folds": 4, "folds": 4, "passes": True} for model in s.FORCE_MODELS}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            s.render_figures(rows, directions, cases, curves, output)
            s.write_report(output, decisions, directions, summary)
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 7)
            self.assertTrue((output / "SUMMARY_ZH.md").is_file())


class FailureRuleTests(unittest.TestCase):
    def test_representative_rules_are_deterministic(self):
        rows = [
            {"model": "V", "fold": "htt_leave_p2", "seed": 2, "failure_type": "false_positive", "episode": "b", "static_alarming_fraction": .8, "false_alarm_starts": 2},
            {"model": "V", "fold": "htt_leave_p1", "seed": 1, "failure_type": "false_positive", "episode": "a", "static_alarming_fraction": .8, "false_alarm_starts": 2},
            {"model": "V", "fold": "htt_leave_p2", "seed": 2, "failure_type": "false_negative", "episode": "d", "event_recall": .5, "missed_gross_events": 2},
            {"model": "V", "fold": "htt_leave_p1", "seed": 1, "failure_type": "false_negative", "episode": "c", "event_recall": .5, "missed_gross_events": 2},
        ]
        selected = s.choose_representatives(rows)
        self.assertEqual([row["episode"] for row in selected], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
