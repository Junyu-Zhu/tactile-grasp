#!/usr/bin/env python3
"""Audit CTA paper experiment provenance without training or inference.

The server is authoritative for run outputs.  This script reads all R22/G1
and R23/G2 run metadata, reads all G2 prediction summaries, inspects G2 NPZ
headers, and loads one real checkpoint per group/fold (44 total: 28 G1 and
16 G2).  It never reads test data, runs a model, or hashes large
checkpoints/data files.  Prediction SHA checks bind existing summary/status
records; they do not recompute every large NPZ digest.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


R22 = "experiments/htt_normalflow/round22_921_g1_joint_frozen"
R23 = "experiments/htt_normalflow/round23_921_g2_force_aux_finetune"
R24 = "experiments/htt_normalflow/round24_921_g3_final_review"
GROUPS = "ABCD"
ROLES = ("fit", "selection", "calibration", "validation")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], *, stdin: str | None = None) -> str:
    p = subprocess.run(cmd, input=stdin, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {p.stderr.strip()}")
    return p.stdout


def remote_probe_source() -> str:
    # Executed with the requested server conda Python.  Loading is sequential
    # and bounded to 16 representative best checkpoints plus four prepared
    # caches (one per fold).  Large artifacts are not re-hashed.
    return r'''
import gc, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

repo=Path(sys.argv[1]); payload=json.loads(sys.stdin.read()); inv=payload["g2"]
g1_inv=payload["g1_inventory"];g1_accept={x["run"]:x for x in payload["g1_acceptance"]["runs"]}
train=Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round22_921_g1_joint_frozen/formal")
pred=Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round23_921_g2_force_aux_finetune/predictions")
status=json.loads((repo/"experiments/htt_normalflow/round23_921_g2_force_aux_finetune/PREDICTION_STATUS.json").read_text())
status_rows={x["run"]:x for x in status["rows"]}

def small_sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

rows=[]; g1_rows=[]; problems=[]; tensors=[]; g1_tensors=[]; datasets=[]; raw_alignment=[]
grid={(x["group"],x["fold"],x["seed"]):x for x in inv["runs"]}
expected={(g,f,s) for g in "ABCD" for f in range(1,5) for s in (20260914,20260915,20260916)}
if set(grid)!=expected:problems.append("inventory_grid_not_48_exact")

for key,item in sorted(grid.items(),key=lambda z:(z[0][1],z[0][2],z[0][0])):
 g,f,s=key; run_id=f"{g}/p{f}_s{s}"; root=train/g/f"p{f}_s{s}"
 paths={n:root/n for n in ("config.json","summary.json","COMMIT.json")}
 pp=pred/g/f"p{f}_s{s}"; paths["prediction_summary.json"]=pp/"SUMMARY.json"; npz=pp/"PREDICTIONS.npz"
 missing=[str(p) for p in [*paths.values(),npz] if not p.is_file()]
 if missing:
  problems.append(f"{run_id}:missing:{missing}");continue
 cfg=json.loads(paths["config.json"].read_text()); sm=json.loads(paths["summary.json"].read_text())
 cm=json.loads(paths["COMMIT.json"].read_text()); ps=json.loads(paths["prediction_summary.json"].read_text())
 st=status_rows.get(run_id,{})
 arr=np.load(npz,allow_pickle=False)
 names=set(arr.files); shapes={k:list(arr[k].shape) for k in arr.files}
 expected_aux=g in "BD"; expected_film=g in "CD"
 role_fields={r:{
   "score":f"{r}_score" in names,
   "aux_pred":f"{r}_aux_force_pred_n" in names,
   "aux_gt":f"{r}_aux_force_gt_n" in names,
   "film_off":f"{r}_film_off_score" in names,
  } for r in ("fit","selection","calibration","validation")}
 checks={
  "identity_matches_inventory": all(cfg.get(k)==item.get(k) for k in ("group","fold","seed","data_sha256","prefix_index_sha256","source_sha256","visual_checkpoint_sha256")),
  "summary_identity_matches_config": sm.get("identity")==cfg,
  "commit_best_matches_index": cm.get("best",{}).get("sha256")==item.get("best",{}).get("sha256"),
  "prediction_binds_best": ps.get("best_commit_sha256")==cm.get("best",{}).get("sha256"),
  "prediction_status_hash_matches": ps.get("prediction_sha256")==st.get("prediction_sha256"),
  "prediction_run_identity": (ps.get("run"),ps.get("group"),ps.get("fold"),ps.get("seed"))==(run_id,g,f,s),
  "strict_model_load_recorded": ps.get("checkpoint_model_loaded_strict") is True,
  "selection_replayed": abs(float(ps.get("selection_replay_abs_error",999))) <= 2e-6,
  "test_not_consumed": ps.get("test_consumed") is False,
  "all_role_scores_present": all(x["score"] for x in role_fields.values()),
  "aux_fields_match_group": all((x["aux_pred"] and x["aux_gt"])==expected_aux for x in role_fields.values()),
  "film_fields_match_group": all(x["film_off"]==expected_film for x in role_fields.values()),
 }
 if not all(checks.values()):problems.append(f"{run_id}:failed:{[k for k,v in checks.items() if not v]}")
 rows.append({"run":run_id,"group":g,"fold":f,"seed":s,"output":str(root),"data":item["data"],
   "data_sha256_recorded":item["data_sha256"],"best_checkpoint":str(root/cm["best"]["path"]),
   "best_checkpoint_sha256_recorded":cm["best"]["sha256"],"prediction":str(npz),
   "prediction_sha256_recorded":ps["prediction_sha256"],"prediction_method":st.get("method"),
   "fine_tuned_suffix_state_sha256":ps.get("fine_tuned_suffix_state_sha256"),
   "normalizer_sha256":cfg.get("normalizer_sha256"),"auxiliary_force_supervision":expected_aux,
   "predicted_force_conditioning":expected_film,"metadata_checks_pass":all(checks.values()),
   "checks":checks,"npz_shapes":shapes,"small_file_sha256":{k:small_sha(v) for k,v in paths.items()}})
 del arr

# R22/G1: all 84 formal metadata records bind inventory -> config/summary ->
# COMMIT best -> final acceptance.  These heads are distinct from R23/G2.
for item in g1_inv["runs"]:
 run_id=item["run"];root=Path(item["output"]);acc=g1_accept.get(run_id,{})
 paths={n:root/n for n in ("config.json","summary.json","COMMIT.json")}
 missing=[str(x) for x in paths.values() if not x.is_file()]
 if missing:problems.append(f"g1:{run_id}:missing:{missing}");continue
 cfg=json.loads(paths["config.json"].read_text());sm=json.loads(paths["summary.json"].read_text());cm=json.loads(paths["COMMIT.json"].read_text())
 checks={"run_identity":(cfg.get("group"),cfg.get("fold"),cfg.get("seed"))==(item["group"],item["fold"],item["seed"]),
  "data_binding":cfg.get("data_sha256")==item.get("data_sha256"),"summary_identity_matches_config":sm.get("identity")==cfg,
  "commit_acceptance_best":cm.get("best",{}).get("sha256")==acc.get("best_sha256"),
  "acceptance_identity":(acc.get("package"),acc.get("group"),acc.get("fold"),acc.get("seed"))==(item["package"],item["group"],item["fold"],item["seed"])}
 if not all(checks.values()):problems.append(f"g1:{run_id}:failed:{[k for k,v in checks.items() if not v]}")
 g1_rows.append({"run":run_id,"package":item["package"],"group":item["group"],"fold":item["fold"],"seed":item["seed"],
  "output":str(root),"data":item["data"],"data_sha256_recorded":item["data_sha256"],
  "best_checkpoint":str(root/cm["best"]["path"]),"best_checkpoint_sha256_recorded":cm["best"]["sha256"],
  "normalizer_sha256":cfg.get("normalizer_sha256"),"metadata_checks_pass":all(checks.values()),"checks":checks,
  "small_file_sha256":{k:small_sha(v) for k,v in paths.items()}})

# Actual tensor identity: fixed seed, every group x fold.  This verifies the
# checkpoint content/identity and group-specific state surface, not just JSON.
for f in range(1,5):
 for g in "ABCD":
  item=grid[g,f,20260914]; root=train/g/f"p{f}_s20260914"; cm=json.loads((root/"COMMIT.json").read_text())
  p=root/cm["best"]["path"]; ck=torch.load(p,map_location="cpu",weights_only=False); ident=ck["identity"]; keys=sorted(ck["model"])
  aux=[k for k in keys if k.startswith("aux.")]; film=[k for k in keys if "film" in k]
  checks={"identity_matches_config":ident==json.loads((root/"config.json").read_text()),
          "best_epoch_matches":int(ck["best_epoch"])==int(cm["best"]["epoch"]),
          "has_trainable_blocks":any(k.startswith("blocks.") for k in keys),
          "aux_state_matches_group":bool(aux)==(g in "BD"),"film_state_matches_group":bool(film)==(g in "CD")}
  if not all(checks.values()):problems.append(f"tensor:{g}/p{f}_s20260914:{checks}")
  tensors.append({"run":f"{g}/p{f}_s20260914","path":str(p),"recorded_sha256":cm["best"]["sha256"],
    "epoch":ck["epoch"],"best_epoch":ck["best_epoch"],"model_tensor_count":len(keys),
    "model_parameter_elements":sum(int(v.numel()) for v in ck["model"].values()),
    "initial_state_sha256":ck.get("initial_state_sha256"),"aux_tensor_keys":aux,"film_tensor_keys":film,"checks":checks})
  del ck;gc.collect()

# G1 tensor sample: every group x fold at fixed seed (7 x 4 = 28).
for item in g1_inv["runs"]:
 if item["seed"]!=20260914:continue
 root=Path(item["output"]);cm=json.loads((root/"COMMIT.json").read_text());p=root/cm["best"]["path"]
 ck=torch.load(p,map_location="cpu",weights_only=False);keys=sorted(ck["model"]);ident=ck["identity"]
 checks={"identity_matches_config":ident==json.loads((root/"config.json").read_text()),
  "best_epoch_matches":int(ck["best_epoch"])==int(cm["best"]["epoch"]),"model_nonempty":bool(keys)}
 if not all(checks.values()):problems.append(f"g1_tensor:{item['run']}:{checks}")
 g1_tensors.append({"run":item["run"],"path":str(p),"recorded_sha256":cm["best"]["sha256"],
  "model_tensor_count":len(keys),"model_parameter_elements":sum(int(v.numel()) for v in ck["model"].values()),
  "model_key_prefixes":sorted({k.split(".",1)[0] for k in keys}),"checks":checks})
 del ck;gc.collect()

# One prepared cache per fold verifies actual schema, role separation and that
# no test role was serialized.  It also records the 195D = visual192+force3
# surface and the explicit y_current supervision target.
for f in range(1,5):
 item=grid["A",f,20260914]; d=torch.load(item["data"],map_location="cpu",weights_only=False)
 prefix=json.loads(Path(item["prefix_index"]).read_text());prefix_by_id={x["episode_id"]:x for x in prefix["entries"]}
 manifest_path=Path(d["provenance"]["immutable_upstream"]["force_prediction_manifest"]["path"])
 force_manifest=json.loads(manifest_path.read_text());force_by_id={x["episode_id"]:x for x in force_manifest["entries"]}
 role_sets={r:set(d["roles"][r]["leakage_group"]) for r in d["roles"]}; overlap={}
 roles=sorted(role_sets)
 for i,a in enumerate(roles):
  for b in roles[i+1:]:overlap[f"{a}|{b}"]=len(role_sets[a]&role_sets[b])
 rec={"fold":f,"path":item["data"],"schema":d.get("schema"),"top_keys":sorted(d),"roles":roles,
  "provenance":d.get("provenance"),
  "has_test": "test" in d["roles"],"role_overlap":overlap,
  "role_shapes":{r:{k:list(v.shape) for k,v in d["roles"][r].items() if hasattr(v,"shape")} for r in roles},
  "role_fields":{r:sorted(d["roles"][r]) for r in roles}}
 rec["checks"]={"schema":rec["schema"]=="round22_e3_current_common_gt_v1","roles_exact":set(roles)==set(("fit","selection","calibration","validation")),"no_test":not rec["has_test"],"no_role_leakage":all(v==0 for v in overlap.values()),"x_195":all(x["x"][1:]==[9,195] for x in rec["role_shapes"].values()),"y_current_3":all(x["y_current"][1:]==[3] for x in rec["role_shapes"].values())}
 if not all(rec["checks"].values()):problems.append(f"data_fold_{f}:{rec['checks']}")
 datasets.append(rec);del d;gc.collect()

 # Fixed first endpoint in every role: follow episode_id mappings, never a
 # basename heuristic.  Recompute current GT from the exact raw NPZ and bind
 # cached predicted-force history to the R10 prediction file.
 d=torch.load(item["data"],map_location="cpu",weights_only=False)
 for role in ("fit","selection","calibration","validation"):
  r=d["roles"][role];i=0;eid=r["episode_id"][i];t=int(r["t"][i]);pe=prefix_by_id[eid];fe=force_by_id[eid]
  raw=np.load(pe["source_path"],allow_pickle=False);fp=np.load(fe["prediction_path"],mmap_mode="r",allow_pickle=False)
  recomputed=np.clip(np.asarray(raw["6d_force"][t,:3],np.float32)-np.asarray(raw["ref_force"][:3],np.float32),-20,20)
  cached=r["y_current"][i].numpy();pred_cached=r["x"][i,:,192:195].numpy();pred_raw=np.asarray(fp[t-8:t+1],np.float32)
  checks={"episode_prefix_exact":pe["episode_id"]==eid,"episode_force_exact":fe["episode_id"]==eid,
   "prefix_source_frames_match":int(pe["frames"])==len(raw["tactile_img"]),"force_prediction_frames_match":int(fe["frames"])==len(fp),
   "t_in_range":13<=t<len(raw["tactile_img"]),"visual_prefix_slice_in_range":0<=t-8 and t<len(raw["tactile_img"]),
   "raw_union_t_minus_13_in_range":0<=t-13,"y_current_matches_raw":bool(np.allclose(cached,recomputed,atol=1e-6,rtol=1e-6)),
   "predicted_force_history_matches":bool(np.allclose(pred_cached,pred_raw,atol=0,rtol=0)),
   "reference_frame_valid":bool(np.isfinite(raw["ref_frame"]).all() and raw["ref_frame"].min()>=0 and raw["ref_frame"].max()<=255)}
  if not all(checks.values()):problems.append(f"raw_alignment:p{f}:{role}:{checks}")
  raw_alignment.append({"fold":f,"role":role,"endpoint_index":i,"episode_id":eid,"t":t,
   "raw_npz":pe["source_path"],"prefix_tensor":pe["path"],"force_prediction":fe["prediction_path"],
   "visual_prefix_indices":[t-8,t],"per_prefix_raw_pair_rule":"[s-5,s] via adapters.window(history=2,stride=5)",
   "raw_image_union":[t-13,t],"reference_frame_key":"ref_frame","reference_frame_shape":list(raw["ref_frame"].shape),
   "force_keys":["6d_force","ref_force"],"raw_force_xyz":np.asarray(raw["6d_force"][t,:3],float).tolist(),
   "ref_force_xyz":np.asarray(raw["ref_force"][:3],float).tolist(),"recomputed_y_current":recomputed.tolist(),
   "cached_y_current":cached.tolist(),"y_current_max_abs_error":float(np.max(np.abs(cached-recomputed))),
   "predicted_force_history_max_abs_error":float(np.max(np.abs(pred_cached-pred_raw))),"checks":checks})
  del raw,fp
 del d;gc.collect()

code_rel=[
 "experiments/htt_normalflow/round22_921_g1_joint_frozen/train_frozen.py",
 "experiments/htt_normalflow/round22_921_g1_joint_frozen/train_e3.py",
 "experiments/htt_normalflow/round22_921_g1_joint_frozen/evaluate_frozen_detection.py",
 "experiments/htt_normalflow/round23_921_g2_force_aux_finetune/predict_e3.py",
 "experiments/htt_normalflow/round23_921_g2_force_aux_finetune/evaluate_e3.py"]
print(json.dumps({"server_repo":str(repo),"runs":rows,"g1_runs":g1_rows,"tensor_samples":tensors,"g1_tensor_samples":g1_tensors,"dataset_samples":datasets,"raw_alignment_samples":raw_alignment,
 "remote_code_sha256":{x:small_sha(repo/x) for x in code_rel},"problems":problems},ensure_ascii=False))
'''


def historical_records(repo: Path) -> list[dict]:
    specs = [
        ("R7", "Sparsh原域开发", "future风险H1/H3/H5", "round7_temporal_fairness_event_warning/SUMMARY_ZH.md",
         "统一t-13..t原始帧依赖；B是Fn/Ft/ratio，C另加有符号XYZ差分；修正R6非等历史，不是完整XYZ上的纯差分消融"),
        ("R8", "Sparsh原域开发", "完整历史/融合/hazard与状态辅助", "round8_force_dynamics_event_time/results/reporting/SUMMARY_ZH.md",
         "复用R7九步历史；独立风险与hazard成对；P3融合+hazard为候选，P4状态预测不胜保持基线"),
        ("R9", "HTT slip开发（future仅复核R8原域）", "当前slip时序力融合与R8 future稳健性", "round9_htt_temporal_force_fusion/PAPER_EVIDENCE.md",
         "HTT主比较不训练future；future结论是R8固定模型原域开发复核，不能与R22/R23混排名"),
        ("R10", "HTT专用force-task/HTT slip开发", "真实力监督适配及旧HTT force任务回退", "round10_htt_force_supervision_adaptation/reviews/REGRESSION_SCOPE.md",
         "old_force_regression明确不是Sparsh原域；不得写成Sparsh力性能保持"),
    ]
    base = repo / "experiments/htt_normalflow"
    out = []
    for rd, domain, task, rel, boundary in specs:
        p = base / rel
        out.append({"round": rd, "domain": domain, "task": task, "source": str(p),
                    "source_sha256": sha(p), "boundary": boundary})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument("--server", default="zjy-4090")
    p.add_argument("--server-repo", default="/home/zjy/document/tactile-grasp/sparsh-force-slip")
    p.add_argument("--server-python", default="/home/zjy/miniconda3/envs/sparsh/bin/python")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args(); repo = a.repo.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    inv_path = repo / R23 / "CHECKPOINT_INDEX.json"
    checkpoint_index = json.loads(inv_path.read_text())
    run_inventory_path = repo / R23 / "G2_RUN_INVENTORY.json"
    run_inventory = json.loads(run_inventory_path.read_text())
    source_rows = {x["run"]: x for x in run_inventory["runs"]}
    merged = []
    for x in checkpoint_index["runs"]:
        merged.append({**source_rows[x["run"]], **x})
    inventory = {**checkpoint_index, "runs": merged}
    encoded = base64.b64encode(remote_probe_source().encode()).decode()
    pycode = f"import base64;exec(base64.b64decode('{encoded}'))"
    remote_shell = " ".join((shlex.quote(a.server_python), "-c", shlex.quote(pycode),
                             shlex.quote(a.server_repo)))
    remote_cmd = ["ssh", a.server, remote_shell]
    g1_inventory_path = repo / R22 / "FROZEN_RUN_INVENTORY.json"
    g1_acceptance_path = repo / R22 / "FROZEN_FORMAL_ACCEPTANCE.json"
    payload = {"g2": inventory, "g1_inventory": json.loads(g1_inventory_path.read_text()),
               "g1_acceptance": json.loads(g1_acceptance_path.read_text())}
    remote = json.loads(run(remote_cmd, stdin=json.dumps(payload)))
    local_code = {rel: sha(repo / rel) for rel in remote["remote_code_sha256"]}
    code_parity = {rel: {"local": local_code[rel], "server": h,
                         "match": local_code[rel] == h}
                   for rel, h in remote["remote_code_sha256"].items()}
    problems = list(remote["problems"])
    problems += [f"local_server_code_mismatch:{k}" for k, v in code_parity.items() if not v["match"]]
    machine = {
        "schema": "cta_paper_source_audit_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"g1_all_run_metadata": 84, "g2_all_run_metadata": 48,
                  "all_run_metadata": 132, "g1_actual_checkpoint_tensor_samples": 28,
                  "g2_actual_checkpoint_tensor_samples": 16, "actual_checkpoint_tensor_samples": 44,
                  "actual_prepared_tensor_samples": 4, "raw_endpoint_alignment_samples": 16,
                  "large_artifacts_rehashed": False,
                  "new_inference_or_training": False, "test_consumed": False},
        "authoritative_server": a.server, "inventory": str(inv_path),
        "inventory_sha256": sha(inv_path), "run_inventory_sha256": sha(run_inventory_path),
        "g1_inventory_sha256": sha(g1_inventory_path), "g1_acceptance_sha256": sha(g1_acceptance_path),
        "code_parity": code_parity,
        "runs": remote["runs"], "g1_runs": remote["g1_runs"],
        "tensor_samples": remote["tensor_samples"], "g1_tensor_samples": remote["g1_tensor_samples"],
        "dataset_samples": remote["dataset_samples"], "raw_alignment_samples": remote["raw_alignment_samples"],
        "history_records": historical_records(repo),
        "problems": problems, "passed": not problems,
    }
    (a.output / "MACHINE_AUDIT.json").write_text(json.dumps(machine, ensure_ascii=False, indent=2) + "\n")
    fields = ["run", "group", "fold", "seed", "output", "data", "data_sha256_recorded",
              "best_checkpoint", "best_checkpoint_sha256_recorded", "prediction",
              "prediction_sha256_recorded", "prediction_method", "fine_tuned_suffix_state_sha256",
              "normalizer_sha256", "auxiliary_force_supervision", "predicted_force_conditioning",
              "metadata_checks_pass"]
    with (a.output / "RUN_SOURCE_MAP.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in machine["runs"]: w.writerow({k: row.get(k) for k in fields})
    g1_fields = ["run", "package", "group", "fold", "seed", "output", "data", "data_sha256_recorded",
                 "best_checkpoint", "best_checkpoint_sha256_recorded", "normalizer_sha256", "metadata_checks_pass"]
    with (a.output / "G1_RUN_SOURCE_MAP.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=g1_fields); w.writeheader()
        for row in machine["g1_runs"]: w.writerow({k: row.get(k) for k in g1_fields})
    with (a.output / "HISTORY_RECORD_MAP.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(machine["history_records"][0])); w.writeheader(); w.writerows(machine["history_records"])
    print(json.dumps({"passed": machine["passed"], "g1_runs": len(machine["g1_runs"]),
                      "g2_runs": len(machine["runs"]),
                      "tensor_samples": len(machine["tensor_samples"])+len(machine["g1_tensor_samples"]),
                      "problems": problems}, ensure_ascii=False))


if __name__ == "__main__":
    main()
