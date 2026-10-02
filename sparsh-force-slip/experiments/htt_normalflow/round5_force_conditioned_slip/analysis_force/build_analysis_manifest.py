#!/usr/bin/env python3
"""Build the fixed Round-5 force-analysis queue; never launches work."""
from __future__ import annotations
import json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
PY="/home/zjy/miniconda3/envs/sparsh/bin/python"
SERVER_REPO=Path("/home/zjy/document/tactile-grasp/sparsh-force-slip")
SERVER_HERE=SERVER_REPO/"experiments/htt_normalflow/round5_force_conditioned_slip/analysis_force"
OUT=Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round5_force_conditioned_slip")
CONTRACT=OUT/"data/contract/contract.json";AUDIT=OUT/"data/CACHE_HASH_AUDIT.json"
INVENTORY=SERVER_REPO/"experiments/htt_normalflow/round5_force_conditioned_slip/RUN_INVENTORY.json"
SOURCE=Path("/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth")
R3=Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round3_mae_slip_adaptation/runs/B/fold_p1/seed_20260914/best.pth")
SEEDS=(20260914,20260915,20260916)

def job(identifier,argv,acceptance,identity,ready):
    return {"id":identifier,"argv":list(map(str,argv)),"acceptance_path":str(acceptance),"log_path":str(OUT/f"analysis_force/logs/{identifier}.log"),"identity":identity,"ready_paths":list(map(str,ready))}

def common(script):return [PY,str(SERVER_HERE/script)]

def build():
    jobs=[]
    for fold in range(1,5):
        for seed in SEEDS:
            run=OUT/f"formal/force/p{fold}_s{seed}";base=["--contract",CONTRACT,"--cache-audit",AUDIT,"--inventory",INVENTORY,"--source-checkpoint",SOURCE,"--variant","adapt","--checkpoint",run/"best.pth","--training-summary",run/"training_summary.json","--fold",f"htt_leave_p{fold}","--seed",seed,"--device","cuda:0"]
            for role in ("calibration","validation"):
                output=OUT/f"analysis_force/htt/p{fold}_s{seed}_{role}"
                jobs.append(job(f"force_p{fold}_s{seed}_{role}",common("analyze_force.py")+["htt",*base,"--role",role,"--output",output],output/"metrics.json",{"format":"round5_htt_force_analysis_v1","variant":"adapt","fold":f"htt_leave_p{fold}","seed":seed,"role":role},[run/"best.pth",run/"training_summary.json"]))
    source_old=OUT/"analysis_force/source/old_val.json"
    jobs.append(job("source_old",common("analyze_force.py")+["source","--features","/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/features/val_features.pt","--source-checkpoint",SOURCE,"--output",source_old],source_old,{"format":"round5_source_force_regression_v1","status":"complete"},[SOURCE]))
    source_adapt=OUT/"analysis_force/source/adapt_all_folds_seeds.json"
    jobs.append(job("source_adapt_all",common("analyze_force.py")+["source-adapt","--all-adapt-root",OUT/"formal/force","--contract",CONTRACT,"--cache-audit",AUDIT,"--inventory",INVENTORY,"--source-checkpoint",SOURCE,"--output",source_adapt,"--device","cuda:0"],source_adapt,{"format":"round5_source_val_adapt_distribution_v1","status":"complete"},[OUT/f"formal/force/p{f}_s{s}/training_summary.json" for f in range(1,5) for s in SEEDS]))
    for variant,force_kind in (("F-old","old"),("F-adapt","adapt")):
        for fold in range(1,5):
            for seed in SEEDS:
                run=OUT/f"formal/slip/{variant}/p{fold}_s{seed}";force=OUT/f"formal/force_predictions/{force_kind}/p{fold}_s{seed}/prediction_manifest.json"
                for role in ("calibration","validation"):
                    output=OUT/f"analysis_force/sensitivity/{variant}/p{fold}_s{seed}_{role}"
                    argv=common("sensitivity.py")+["--contract",CONTRACT,"--cache-audit",AUDIT,"--inventory",INVENTORY,"--source-checkpoint",SOURCE,"--base-slip-checkpoint",Path(str(R3).replace("fold_p1/seed_20260914",f"fold_p{fold}/seed_{seed}")),"--checkpoint",run/"best.pth","--training-summary",run/"training_summary.json","--force-predictions",force,"--variant",variant,"--fold",f"htt_leave_p{fold}","--seed",seed,"--role",role,"--output",output,"--device","cuda:0"]
                    jobs.append(job(f"sensitivity_{variant}_p{fold}_s{seed}_{role}",argv,output/"metrics.json",{"format":"round5_force_input_sensitivity_v1","variant":variant,"fold":f"htt_leave_p{fold}","seed":seed,"role":role},[run/"training_summary.json",force]))
    rep={"fold":1,"seed":20260914};e2e=[]
    for variant in ("mae-b","V","F-old","F-adapt","future-full"):
        output=OUT/f"analysis_force/e2e/{variant}.json";argv=common("benchmark_e2e.py")+["--contract",CONTRACT,"--cache-audit",AUDIT,"--inventory",INVENTORY,"--source-checkpoint",SOURCE,"--base-slip-checkpoint",R3,"--variant",variant,"--output",output,"--device","cuda:0"]
        ready=[R3]
        slip_variant="F-adapt" if variant=="future-full" else variant
        if variant!="mae-b":
            run=OUT/f"formal/slip/{slip_variant}/p1_s20260914";argv += ["--slip-checkpoint",run/"best.pth","--slip-training-summary",run/"training_summary.json"];ready += [run/"training_summary.json"]
        if variant in ("F-adapt","future-full"):
            run=OUT/"formal/force/p1_s20260914";argv += ["--force-checkpoint",run/"best.pth","--force-training-summary",run/"training_summary.json"];ready += [run/"training_summary.json"]
        if variant=="future-full":
            run=OUT/"formal/htt_future/full_state/seed_20260914";argv += ["--future-checkpoint",run/"best.pth","--future-summary",run/"summary.json","--future-inventory",INVENTORY];ready += [run/"summary.json"]
        jobs.append(job(f"e2e_{variant}",argv,output,{"format":"round5_e2e_latency_v1","variant":variant,"fold":"htt_leave_p1","seed":20260914},ready))
    for variant in ("V","F-old","F-adapt"):
        run=OUT/f"formal/slip/{variant}/p1_s20260914";output=OUT/f"analysis_force/failures/{variant}_p1_s20260914"
        jobs.append(job(f"failures_{variant}",common("export_failures.py")+["--inventory",INVENTORY,"--predictions",run/"predictions_validation.csv","--training-summary",run/"training_summary.json","--contract",CONTRACT,"--fold","htt_leave_p1","--seed",20260914,"--variant",variant,"--role","validation","--output",output],output/"manifest.json",{"format":"round5_failure_images_v1","variant":variant,"fold":"htt_leave_p1","seed":20260914,"role":"validation"},[run/"training_summary.json",run/"predictions_validation.csv"]))
    return {"format":"round5_force_analysis_queue_v1","status":"fixed","jobs":jobs,"job_count":len(jobs),"gpu_execution":"single sequential GPU; root-owned scheduling","builder_launches_processes":False}

if __name__=="__main__":
    output=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/"ANALYSIS_MANIFEST.json"
    output.write_text(json.dumps(build(),indent=2)+"\n");print(json.dumps({"status":"built_no_process_started","jobs":len(build()["jobs"]),"output":str(output.resolve())},indent=2))
