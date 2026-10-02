from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import json
from pathlib import Path

import torch

P = Path(__file__).with_name("future_pipeline.py")
S = importlib.util.spec_from_file_location("r5future", P)
m = importlib.util.module_from_spec(S)
sys.modules["r5future"] = m
S.loader.exec_module(m)


def source_payload(samples=range(16)):
    samples = list(samples)
    n = len(samples)
    force = torch.tensor([[float(t), 2.0 * float(t), 3.0 * float(t)] for t in samples])
    return {
        "z": torch.randn(n, 4), "future_slip": torch.zeros(n, 3), "force_pred_n": force,
        "slip_probs": torch.tensor([[.8, .2]] * n),
        "metadata": [{"dataset": "d", "trajectory": "e", "sample": t} for t in samples],
        "horizons": [1, 3, 5], "aux": torch.randn(n, 7),
    }


def htt_payload(formal=True):
    episodes = []
    for k in range(15):
        t = torch.arange(0, 31)
        stage = torch.zeros(31)
        if k < 5:
            stage[22] = 2
        episodes.append({"episode_id": f"tr{k}", "role": "train", "t": t, "z": torch.randn(31, 4), "p_slip": torch.rand(31), "base_p_slip": torch.rand(31), "force_pred_n": torch.randn(31, 3), "stage": stage})
    for k in range(2):
        t = torch.arange(0, 31)
        episodes.append({"episode_id": f"v{k}", "role": "validation", "t": t, "z": torch.randn(31, 4), "p_slip": torch.rand(31), "base_p_slip": torch.rand(31), "force_pred_n": torch.randn(31, 3), "stage": torch.zeros(31)})
    return {"schema": "round5_future_upstream_v1", "fold": "htt_leave_p1", "upstream_seed": 20260914, "upstream_force": "F-adapt", "upstream_slip": "R5-F-adapt-conditioned-slip", "condition_normalization": {"ratio_epsilon_n": 0.25}, "formal": formal, "episodes": episodes}


def nested_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


class FuturePipelineTests(unittest.TestCase):
    def test_source_cache_uses_predicted_exact_delta_and_common_t10(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); src = d / "old.pt"; out = d / "new.pt"; torch.save(source_payload(), src)
            audit = m.build_source_cache(src, out, "train"); p = torch.load(out, weights_only=False)
            self.assertEqual(audit["audit"]["minimum_sample"], 10)
            self.assertEqual(p["metadata"][0]["sample"], 10)
            self.assertTrue(torch.equal(p["predicted_delta_force_xyz"][0], torch.tensor([5., 10., 15.])))
            self.assertTrue(p["audit"]["old_delta_source"].startswith("ground_truth"))
            self.assertEqual(m.source_input(p, "z_p_slip_force_pred_delta").shape[1], 11)

    def test_source_cache_rejects_missing_lag_without_interpolation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); src = d / "old.pt"; out = d / "new.pt"
            torch.save(source_payload([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]), src)
            a = m.build_source_cache(src, out, "val")
            self.assertEqual(a["audit"]["rejected_missing_exact_lag"], 1)
            p = torch.load(out, weights_only=False)
            self.assertNotIn(16, [x["sample"] for x in p["metadata"]])

    def test_htt_support_is_train_only_and_requires_complete_recursive_history(self):
        a = m.htt_support_audit(htt_payload())
        self.assertEqual(a["minimum_history_endpoint"], 13)
        self.assertEqual(a["selected_horizon"], 8)
        h8 = a["candidates"][0]
        self.assertEqual(h8["positive_onset_episodes"], 5)
        self.assertEqual(h8["negative_episodes"], 15)
        self.assertEqual(h8["negative_frames"], 105)

    def test_support_never_counts_static_recovery_after_first_gross(self):
        t = torch.arange(0, 50); stage = torch.zeros(50); stage[20] = 2
        p = {"episodes": [{"episode_id": "e", "role": "train", "t": t, "stage": stage}]}
        a = m.htt_support_audit(p); h8 = a["candidates"][0]
        self.assertEqual(h8["positive_frames"], 7)  # eligible pre-onset endpoints 13..19
        self.assertEqual(h8["negative_frames"], 0)

    def test_full_state_target_is_joint_h1_h3_and_selected_horizon(self):
        payload = htt_payload(False)
        x, _, residual = m._htt_examples(payload, 8, "train", "full_state")
        z_dim = payload["episodes"][0]["z"].shape[1]
        self.assertEqual(residual.shape[1], 3 * z_dim)
        self.assertEqual(x.shape[2], z_dim + 1 + 5)

    def test_formal_provenance_rejects_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.pt"; torch.save(htt_payload(formal=False), p)
            with self.assertRaisesRegex(ValueError, "smoke"):
                m.canonical_htt_payload(p, require_formal=True)

    def test_assembler_binds_hashes_and_excludes_test(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); roles={"htt_leave_p1":"train","htt_leave_p2":"test","htt_leave_p3":"train","htt_leave_p4":"calibration"}
            token=d/"tokens.npy";label=d/"labels.npy";predfile=d/"pred.npz"
            __import__("numpy").save(token,torch.randn(20,3,4).numpy());__import__("numpy").save(label,torch.zeros(20).numpy())
            __import__("numpy").savez(predfile,frame_index=__import__("numpy").arange(20),force_pred_n=__import__("numpy").zeros((20,3),dtype="float32"),p_slip=__import__("numpy").zeros(20,dtype="float32"),base_p_slip=__import__("numpy").zeros(20,dtype="float32"))
            force_parent=d/"force_manifest.json";force_parent.write_text(json.dumps({"status":"complete","format":"round5_force_predictions_v1","variant":"adapt","fold":"htt_leave_p1","seed":20260914,"formal":False}))
            fused_config={"variant":"F-adapt","fold":"htt_leave_p1","seed":20260914,"smoke":True}
            fused_parent=d/"fused.pth";torch.save({"format":"round5_force_conditioned_slip_v1","config":fused_config},fused_parent)
            base_parent=d/"base.pth";torch.save({"format":"round3_mae_slip_branch_v1","config":{"init":"fresh","fold":"htt_leave_p1","seed":20260914,"smoke":False}},base_parent)
            contract={"format":"round5_htt_mae_tokens_targets_v1","status":"complete","entries":[{"episode_id":"e","task":"slip","roles_by_fold":roles,"frames":20,"token_path":str(token),"token_sha256":m.sha256(token),"label_path":str(label),"label_sha256":m.sha256(label)}]}
            cp=d/"contract.json";cp.write_text(json.dumps(contract))
            manifest={"format":"round5_fadapt_contact_predictions_v1","status":"complete","formal":False,"fold":"htt_leave_p1","seed":20260914,"token_cache_sha256":m.sha256(cp),"condition_normalization":{"ratio_epsilon_n":0.25},"force_prediction_manifest":str(force_parent),"force_prediction_manifest_sha256":m.sha256(force_parent),"fused_slip_checkpoint":str(fused_parent),"fused_slip_checkpoint_sha256":m.sha256(fused_parent),"fused_slip_config_sha256":m.identity_hash(fused_config),"base_slip_checkpoint":str(base_parent),"base_slip_checkpoint_sha256":m.sha256(base_parent),"entries":[{"episode_id":"e","roles_by_fold":roles,"frames":20,"path":str(predfile),"sha256":m.sha256(predfile)}]}
            mp=d/"manifest.json";mp.write_text(json.dumps(manifest));out=d/"out.pt"
            with self.assertRaisesRegex(ValueError,"smoke"):m.assemble_htt_upstream(cp,mp,out)
            result=m.assemble_htt_upstream(cp,mp,out,allow_smoke=True);payload=torch.load(out,weights_only=False)
            self.assertEqual(result["status"],"complete");self.assertTrue(result["smoke"])
            self.assertEqual(result["episode_count"],1);self.assertFalse(payload["test_role_consumed"]);self.assertEqual(tuple(payload["episodes"][0]["z"].shape),(20,4))
            force_parent.write_text("{}")
            with self.assertRaisesRegex(ValueError,"parent"):
                m.assemble_htt_upstream(cp,mp,d/"tampered.pt",allow_smoke=True)

    def test_inventory_is_exactly_18_runs(self):
        with tempfile.TemporaryDirectory() as d:
            p = m.inventory(Path(d) / "i.json")
            self.assertEqual(p["formal_runs"], 18)
            self.assertEqual(sum(x["domain"] == "source" for x in p["runs"]), 9)
            self.assertEqual(sum(x["domain"] == "htt" for x in p["runs"]), 9)

    def test_source_interrupted_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); old = d / "old.pt"; torch.save(source_payload(), old)
            tr = d / "tr.pt"; va = d / "va.pt"
            m.build_source_cache(old, tr, "train"); m.build_source_cache(old, va, "val")
            m.train_source(tr, va, d / "continuous", "z_p_slip", 20260914, "cpu", 2)
            m.train_source(tr, va, d / "resumed", "z_p_slip", 20260914, "cpu", 2, interrupt_after_epoch=1)
            interrupted = json.loads((d / "resumed/summary.json").read_text())
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["interrupted_after_epoch"], 1)
            self.assertEqual(interrupted["target_epochs"], 2)
            m.train_source(tr, va, d / "resumed", "z_p_slip", 20260914, "cpu", 2, resume=True)
            continuous = torch.load(d / "continuous/latest.pth", weights_only=False)
            resumed = torch.load(d / "resumed/latest.pth", weights_only=False)
            a, b = continuous["model_state"], resumed["model_state"]
            self.assertEqual(a.keys(), b.keys()); self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            self.assertTrue(nested_equal(continuous["optimizer_state"], resumed["optimizer_state"]))
            self.assertEqual(continuous["history"], resumed["history"])
            self.assertEqual(torch.load(d / "resumed/latest.pth", weights_only=False)["run_config"]["effective_epoch_budget"], 2)
            config = torch.load(d / "resumed/latest.pth", weights_only=False)["run_config"]
            self.assertEqual(config["model_logit_semantics"], "stable_no_future_slip")
            self.assertEqual(config["reported_probability_semantics"], "future_any_slip=1-sigmoid(stable_logit)")
            self.assertEqual(len(config["code_bundle_sha256"]), 64)

    def test_htt_interrupted_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); up = d / "up.pt"; torch.save(htt_payload(formal=False), up)
            support = d / "support.json"; m.atomic_json(support, m.htt_support_audit(htt_payload(False)))
            m.train_htt(up, support, d / "continuous", "full_state", 20260914, "cpu", 2, False)
            m.train_htt(up, support, d / "resumed", "full_state", 20260914, "cpu", 2, False, interrupt_after_epoch=1)
            interrupted = json.loads((d / "resumed/summary.json").read_text())
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["interrupted_after_epoch"], 1)
            self.assertEqual(interrupted["target_epochs"], 2)
            m.train_htt(up, support, d / "resumed", "full_state", 20260914, "cpu", 2, False, resume=True)
            continuous = torch.load(d / "continuous/latest.pth", weights_only=False)
            resumed = torch.load(d / "resumed/latest.pth", weights_only=False)
            a, b = continuous["model_state"], resumed["model_state"]
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            self.assertTrue(nested_equal(continuous["optimizer_state"], resumed["optimizer_state"]))
            self.assertEqual(continuous["history"], resumed["history"])
            summary = __import__("json").loads((d / "resumed/summary.json").read_text())
            self.assertTrue(summary["smoke"]); self.assertFalse(summary["formal"])
            self.assertIn("sha256", summary["artifacts"]["latest"])
            best = torch.load(d / "resumed/best.pth", weights_only=False)
            self.assertEqual(best["run_config"]["state_prediction_horizons"], [1, 3, 8])
            self.assertEqual(best["run_config"]["state_output_dim"], 12)

    def test_nonresume_refuses_to_overwrite_run_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); old = d / "old.pt"; torch.save(source_payload(), old)
            tr = d / "tr.pt"; va = d / "va.pt"; out = d / "run"
            m.build_source_cache(old, tr, "train"); m.build_source_cache(old, va, "val")
            m.train_source(tr, va, out, "z_p_slip", 20260914, "cpu", 1)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                m.train_source(tr, va, out, "z_p_slip", 20260914, "cpu", 1)

    def test_source_cli_interrupt_then_resume_keeps_two_epoch_budget(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); old = d / "old.pt"; torch.save(source_payload(), old)
            tr = d / "tr.pt"; va = d / "va.pt"; out = d / "smoke_cli"
            m.build_source_cache(old, tr, "train"); m.build_source_cache(old, va, "val")
            common = ["train-source", "--train-cache", str(tr), "--validation-cache", str(va),
                      "--variant", "z_p_slip", "--seed", "20260914", "--output", str(out),
                      "--protocol-sha256", m.protocol_sha256(), "--smoke-epochs", "2", "--device", "cpu"]
            self.assertEqual(m.main([*common, "--interrupt-after-epoch", "1"]), 0)
            self.assertEqual(json.loads((out / "summary.json").read_text())["target_epochs"], 2)
            self.assertEqual(m.main([*common, "--resume"]), 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(summary["history"]), 2)


if __name__ == "__main__":
    unittest.main()
