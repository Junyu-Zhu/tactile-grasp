import importlib.util,json,tempfile,unittest
from pathlib import Path

P=Path(__file__).with_name("plot_results.py")
S=importlib.util.spec_from_file_location("r5_future_plots",P);plot=importlib.util.module_from_spec(S);S.loader.exec_module(plot)


class PlotTests(unittest.TestCase):
    def test_smoke_rejection_and_all_figures(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/"source.json";htt=d/"htt.json";out=d/"figures"
            learned=[]
            for variant in ("z_p_slip","z_p_slip_force","z_p_slip_force_pred_delta"):
                for seed in (20260914,20260915,20260916):
                    for h in (1,3,5):
                        for method in ("raw","raw_mul_pslip_fixed"):
                            learned.append({"variant":variant,"seed":seed,"horizon":h,"method":method,"validation":{"average_precision":.6,"brier":.2}})
            source.write_text(json.dumps({"status":"complete","formal":False,"domain":"source","results":learned}))
            with self.assertRaisesRegex(ValueError,"nonformal"):plot.load_evaluation(source,"source")
            source_payload=plot.load_evaluation(source,"source",True);plot.plot_source(source_payload,out)
            results=[]
            for variant in ("risk","base","full_state"):
                for seed in (20260914,20260915,20260916):
                    for method in ("raw","raw_mul_pslip_fixed"):
                        for op in ("fixed_0.5","max_ba"):
                            results.append({"variant":variant,"seed":seed,"method":method,"operating_point":op,"validation":{"fpr":.1,"recall":.7}})
            metric=lambda h:{"horizon":h,"mse_raw":.2,"variance_replication_ratio":.5}
            states=[]
            for seed in (20260914,20260915,20260916):
                row={"seed":seed,"learned_full_state":[metric(1),metric(3),metric("H")]}
                if seed==20260914:row.update({"persistence_zero_residual":[metric(1),metric(3),metric("H")],"train_only_ridge":[metric(1),metric(3),metric("H")]})
                states.append(row)
            htt_payload={"status":"complete","formal":True,"domain":"htt","horizon":8,"results":results,"state_evaluation":states}
            plot.plot_htt(htt_payload,out);plot.plot_state(htt_payload,out)
            self.assertEqual({p.name for p in out.iterdir()},{"source_ap_brier_by_horizon_seed.svg","source_ap_brier_by_horizon_seed.png","htt_validation_fpr_recall.svg","htt_validation_fpr_recall.png","htt_state_error_variance_by_horizon_seed.svg","htt_state_error_variance_by_horizon_seed.png"})


if __name__=="__main__":unittest.main()
