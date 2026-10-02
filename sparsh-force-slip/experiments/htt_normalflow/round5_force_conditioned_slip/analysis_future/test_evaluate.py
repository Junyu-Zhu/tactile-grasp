import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch

P=Path(__file__).with_name("evaluate.py")
S=importlib.util.spec_from_file_location("r5_future_eval",P);e=importlib.util.module_from_spec(S);S.loader.exec_module(e)
RP=Path(__file__).with_name("render_reports.py")
RS=importlib.util.spec_from_file_location("r5_future_reports",RP);reports=importlib.util.module_from_spec(RS);RS.loader.exec_module(reports)


class EvaluationTests(unittest.TestCase):
    def test_formal_evaluator_source_snapshot_is_immutable(self):
        self.assertEqual(e.sha256(P),"433cbbaf6de68115e862ab9276151f8b986582594a71a267be51fb44f85ea8f2")

    def test_calibration_includes_global_never_and_tie_rules(self):
        y=np.array([0,0,1,1]);s=np.array([.2,.2,.2,.2]);chosen=e.calibrate(y,s)
        self.assertGreater(chosen["max_ba"],1.)
        self.assertGreater(chosen["fpr_0.01"],1.)

    def test_fixed_multiplicative_gate_is_literal_product(self):
        future=np.array([.8,.2]);slip=np.array([.25,.5])
        self.assertTrue(np.array_equal(future*slip,np.array([.2,.1])))

    def test_cluster_bootstrap_is_deterministic_and_complete_cluster(self):
        rows=[]
        for cluster in ("a","b","c"):
            rows += [{"cluster":cluster,"target":0,"score":.1},{"cluster":cluster,"target":1,"score":.9}]
        self.assertEqual(e.cluster_bootstrap(rows,"score",.5,7,20),e.cluster_bootstrap(rows,"score",.5,7,20))
        self.assertEqual(e.cluster_bootstrap(rows,"score",.5,7,20)["balanced_accuracy"]["valid_replicates"],20)

    def test_event_statuses_distinguish_early_late_and_miss(self):
        rows=[]
        for eid,alarm in (("early",17),("late",20),("miss",None)):
            for t in range(13,29):
                rows.append({"episode_id":eid,"onset":20,"t":t,"eligible":t<20,"target":int(12<=t<20),"score":float(t==alarm)})
        result=e.event_metrics(rows,"score",.5,8)
        self.assertEqual({r["episode_id"]:r["status"] for r in result["records"]},{"early":"early","late":"late","miss":"miss"})
        self.assertEqual(next(r for r in result["records"] if r["episode_id"]=="early")["lead_frames"],3)

    def test_event_without_complete_pre_onset_window_is_censored(self):
        rows=[{"episode_id":"short","onset":14,"t":t,"eligible":False,"target":None,"score":0.0} for t in range(13,20)]
        result=e.event_metrics(rows,"score",.5,8)
        self.assertEqual(result["events_censored"],1);self.assertEqual(result["events_eligible"],0);self.assertEqual(result["miss"],0)

    def test_source_history_requires_exact_four_frames(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/"source.pt"
            metadata=[{"dataset":"d","trajectory":"e","sample":t} for t in (7,8,9,10,12)]
            torch.save({"metadata":metadata,"slip_probs":torch.tensor([[0.,t/20] for t in (7,8,9,10,12)])},source)
            cache={"source_path":str(source),"source_sha256":e.sha256(source),"metadata":[metadata[3],metadata[4]]}
            history=e.exact_source_history(cache)
            self.assertIn(0,history);self.assertNotIn(1,history)
            self.assertAlmostEqual(history[0],np.mean([7,8,9,10])/20)

    def test_state_normalizer_checks_each_horizon_block(self):
        target=torch.arange(1,9,dtype=torch.float32)[:,None]*torch.arange(1,13,dtype=torch.float32)[None,:]
        checkpoint={"normalization":{"state_residual_mean":target.mean(0,keepdim=True),"state_residual_std":target.std(0,keepdim=True).clamp_min(1e-6)}}
        result=e.check_state_normalizer(list(target),checkpoint,8)
        self.assertEqual(result["output_dim"],12);self.assertFalse(result["copied_horizon_std"])
        state=e.state_metrics(target.numpy(),(target*0.5).numpy(),4)
        self.assertAlmostEqual(state[0]["variance_replication_ratio"],.25)
        bad={"normalization":dict(checkpoint["normalization"])};bad["normalization"]["state_residual_std"]=checkpoint["normalization"]["state_residual_std"].clone()
        bad["normalization"]["state_residual_std"][0,4:8]=bad["normalization"]["state_residual_std"][0,:4]
        with self.assertRaisesRegex(ValueError,"normalizer"):e.check_state_normalizer(list(target),bad,8)

    def test_htt_report_names_operating_points_and_event_limits(self):
        metric={"balanced_accuracy":.6,"fpr":.1,"recall":.5,"average_precision":.7,"brier":.2}
        events={"events_eligible":2,"events_total":12,"events_censored":10}
        results=[{"variant":"risk","seed":20260914,"method":"raw_mul_pslip_fixed","operating_point":"fixed_0.5","validation_observed_no_alarm":True,"validation":metric,"validation_events":events},
                 {"variant":"risk","seed":20260914,"method":"raw","operating_point":"max_ba","validation_observed_no_alarm":False,"validation":metric,"validation_events":events}]
        aggregate={"variant":"risk","method":"raw","operating_point":"fpr_0.05","validation":{key:{"mean":value,"std":0.} for key,value in metric.items()}}
        report=reports.render_report({"domain":"htt","formal":True,"evaluation_protocol_sha256":"x","horizon":8,"results":results,"seed_aggregates":[aggregate]})
        self.assertIn("max_ba（仅由 calibration 选择）",report)
        self.assertIn("2/12",report);self.assertIn("H=8 标签窗口边界",report)
        self.assertIn("1/1 个 learned run",report);self.assertIn("calibration operating point",report)

    def test_formal_guard_rejects_smoke_and_verifies_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);run=d/"run";run.mkdir();bp=run/"best.pth";lp=run/"latest.pth"
            config={"domain":"htt","protocol_sha256":"x","code_bundle_sha256":e.r5.code_bundle_sha256()};identity="identity"
            torch.save({"run_config":config,"run_identity_sha256":identity},bp);torch.save({"run_config":config,"run_identity_sha256":identity},lp)
            summary={"status":"complete","variant":"risk","seed":20260914,"smoke":True,"formal":False,"run_config":config,"run_identity_sha256":identity,"artifacts":{"best":{"path":str(bp),"sha256":e.sha256(bp)},"latest":{"path":str(lp),"sha256":e.sha256(lp)}}}
            (run/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,"refuses"):e.verified_run(run,"htt","risk",20260914,"x")
            summary.update({"smoke":False,"formal":True});(run/"summary.json").write_text(json.dumps(summary))
            argv=["python","future_pipeline.py","train-htt","--variant","risk","--seed","20260914","--output",str(run)]
            (run/"summary.json.scheduler_receipt.json").write_text(json.dumps({"argv":argv,"input_sha256":{}}))
            inventory={"jobs":[{"acceptance_path":str(run/"summary.json"),"argv":argv}]}
            checked,_=e.verified_run(run,"htt","risk",20260914,"x",inventory=inventory)
            self.assertIn("scheduler_receipt_sha256",checked["_verified_provenance"])
            lp.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"hash"):e.verified_run(run,"htt","risk",20260914,"x",inventory=inventory)

    def test_formal_guard_rejects_receipt_input_drift_and_inventory_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);run=d/"run";run.mkdir();bp=run/"best.pth";lp=run/"latest.pth";parent=d/"input.pt";parent.write_bytes(b"original")
            config={"domain":"source","protocol_sha256":"x","code_bundle_sha256":e.r5.code_bundle_sha256()};identity="identity"
            for p in (bp,lp):torch.save({"run_config":config,"run_identity_sha256":identity},p)
            summary={"status":"complete","variant":"z_p_slip","seed":20260914,"smoke":False,"formal":True,"run_config":config,"run_identity_sha256":identity,"artifacts":{"best":{"path":str(bp),"sha256":e.sha256(bp)},"latest":{"path":str(lp),"sha256":e.sha256(lp)}}}
            (run/"summary.json").write_text(json.dumps(summary))
            argv=["python","future_pipeline.py","train-source","--variant","z_p_slip","--seed","20260914","--output",str(run)]
            (run/"summary.json.scheduler_receipt.json").write_text(json.dumps({"argv":argv,"input_sha256":{str(parent):e.sha256(parent)}}))
            inventory={"jobs":[{"acceptance_path":str(run/"summary.json"),"argv":argv}]}
            parent.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"input drift"):e.verified_run(run,"source","z_p_slip",20260914,"x",inventory=inventory)
            parent.write_bytes(b"original")
            inventory["jobs"][0]["argv"]=argv+["--resume"]
            with self.assertRaisesRegex(ValueError,"frozen inventory"):e.verified_run(run,"source","z_p_slip",20260914,"x",inventory=inventory)


if __name__=="__main__":unittest.main()
