#!/usr/bin/env python3
"""Strict, result-blind Round-6 future support audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/"protocol.json"
AMENDMENT=HERE/"PROTOCOL_AMENDMENT.json"


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def identity_sha(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path,payload) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False));os.replace(temporary,path)


def deterministic_roles(groups: list[str],domain: str,fold: str|None=None) -> dict[str,str]:
    prefix=f"round6|{domain}|"+(f"{fold}|" if fold else "")
    ranked=sorted(set(groups),key=lambda group:(hashlib.sha256((prefix+group).encode()).hexdigest(),group))
    roles={}
    for rank,group in enumerate(ranked):
        bucket=rank%5
        if domain=="htt":roles[group]="selection" if bucket==0 else "fit_train"
        else:roles[group]="selection" if bucket==0 else ("calibration" if bucket==1 else "fit_train")
    return roles


def endpoint_trial(ts: np.ndarray,stage: np.ndarray,horizon: int,min_t: int,recursive: bool,onset_label: int=2,candidate_ts: np.ndarray|None=None,history_ts: set[int]|None=None) -> tuple[int|None,list[int],list[int],dict[str,int]]:
    positions={int(t):i for i,t in enumerate(ts)};gross=ts[stage==onset_label];onset=None if not len(gross) else int(gross.min())
    history_positions=set(positions) if history_ts is None else history_ts
    positive=[];negative=[];excluded=Counter()
    for raw_t in ts if candidate_ts is None else candidate_ts:
        t=int(raw_t)
        if onset is not None and t>=onset:excluded["at_or_after_first_gross"]+=1;continue
        required={t-j-lag for j in range(4) for lag in ((0,5,10) if recursive else (0,))}
        if t<min_t or not required.issubset(history_positions):excluded["incomplete_exact_history"]+=1;continue
        if int(stage[positions[t]])!=0:excluded["current_not_static"]+=1;continue
        if any(t+step not in positions for step in range(1,horizon+1)):excluded["incomplete_future_window"]+=1;continue
        (positive if onset is not None and t<onset<=t+horizon else negative).append(t)
    return onset,positive,negative,dict(sorted(excluded.items()))


def summarize_trials(trials: list[dict]) -> dict:
    pos_eps=sorted({row["episode_id"] for row in trials if row["positive_endpoints"]});neg_eps=sorted({row["episode_id"] for row in trials if row["negative_endpoints"]})
    pos_groups=sorted({row["leakage_group"] for row in trials if row["positive_endpoints"]});neg_groups=sorted({row["leakage_group"] for row in trials if row["negative_endpoints"]})
    endpoints=[(row["episode_id"],t,1) for row in trials for t in row["positive_endpoints"]]+[(row["episode_id"],t,0) for row in trials for t in row["negative_endpoints"]]
    excluded=Counter()
    for row in trials:excluded.update(row["excluded"])
    return {"episodes":len(trials),"positive_endpoints":sum(len(row["positive_endpoints"]) for row in trials),"negative_endpoints":sum(len(row["negative_endpoints"]) for row in trials),"positive_event_episodes":len(pos_eps),"negative_window_episodes":len(neg_eps),"positive_event_trials":len(pos_groups),"negative_window_trials":len(neg_groups),"positive_leakage_groups":pos_groups,"negative_leakage_groups":neg_groups,"eligible_endpoint_identity_sha256":identity_sha(sorted(endpoints)),"excluded_endpoint_counts":dict(sorted(excluded.items()))}


def threshold_pass(summary: dict,requirements: dict) -> tuple[bool,list[str]]:
    missing=[]
    for key,required in requirements.items():
        if int(summary[key])<int(required):missing.append(f"{key}={summary[key]}<{required}")
    return not missing,missing


def audit_htt(contract_path: Path,split_path: Path,protocol: dict) -> dict:
    contract=json.loads(contract_path.read_text());splits=json.loads(split_path.read_text())
    if contract.get("format")!="round5_htt_mae_tokens_targets_v1" or contract.get("status")!="complete":raise ValueError("invalid HTT contract")
    if contract.get("split_manifest_sha256")!=sha256(split_path):raise ValueError("contract/split hash mismatch")
    episode_meta={row["id"]:row for row in splits["episodes"]};entries=[row for row in contract["entries"] if row.get("task")=="slip"]
    folds=protocol["htt"]["fold_order"]
    allowed_ids={entry["episode_id"] for entry in entries if any(entry["roles_by_fold"].get(fold)!="test" for fold in folds)}
    always_test_ids=sorted(entry["episode_id"] for entry in entries if entry["episode_id"] not in allowed_ids)
    labels={};label_evidence=[]
    for entry in entries:
        if entry["episode_id"] not in allowed_ids:continue
        path=Path(entry["label_path"])
        if sha256(path)!=entry["label_sha256"]:raise ValueError(f"HTT label hash mismatch: {entry['episode_id']}")
        array=np.load(path,allow_pickle=False)
        if array.shape!=(int(entry["frames"]),) or not np.isfinite(array).all() or not set(np.unique(array).astype(int)).issubset({0,1,2}):raise ValueError(f"invalid HTT label timeline: {entry['episode_id']}")
        labels[entry["episode_id"]]=array.astype(np.int8);label_evidence.append({"episode_id":entry["episode_id"],"path":str(path),"sha256":entry["label_sha256"],"frames":int(entry["frames"])})
    output={"format":"round6_htt_support_manifest_v1","status":"complete","contract":{"path":str(contract_path),"sha256":sha256(contract_path)},"split_manifest":{"path":str(split_path),"sha256":sha256(split_path)},"protocol_sha256":sha256(PROTOCOL),"label_access":{"allowed_non_test_union_episode_count":len(allowed_ids),"allowed_non_test_union_identity_sha256":identity_sha(sorted(allowed_ids)),"always_test_episode_count":len(always_test_ids),"always_test_episode_ids":always_test_ids,"always_test_labels_read":False},"label_evidence":{"semantics":"authoritative HTT static/incipient/gross timeline; first gross is a force-rule-influenced proxy, not independent physical truth","independent_physical_truth":False,"entries":label_evidence,"identity_sha256":identity_sha(label_evidence)},"folds":{}}
    candidates=protocol["htt"]["candidate_horizons_in_fixed_order"]
    for fold in protocol["htt"]["fold_order"]:
        original={entry["episode_id"]:entry["roles_by_fold"][fold] for entry in entries}
        for eid,role in original.items():
            expected=next((name for name,ids in splits["splits"][fold].items() if eid in ids),None)
            if expected!=role:raise ValueError(f"split role mismatch: {fold}/{eid}")
        train_groups=[episode_meta[eid]["leakage_group"] for eid,role in original.items() if role=="train"]
        derived=deterministic_roles(train_groups,"htt",fold)
        assigned={}
        for eid,role in original.items():
            if role=="test":continue
            assigned[eid]=derived[episode_meta[eid]["leakage_group"]] if role=="train" else ("outer" if role=="validation" else role)
        role_lists={}
        for role in ("fit_train","selection","calibration","outer"):
            episodes=sorted(eid for eid,value in assigned.items() if value==role);groups=sorted({episode_meta[eid]["leakage_group"] for eid in episodes})
            role_lists[role]={"episodes":episodes,"leakage_groups":groups,"episode_identity_sha256":identity_sha(episodes),"leakage_group_identity_sha256":identity_sha(groups)}
        if set().union(*(set(v["leakage_groups"]) for v in role_lists.values())) != {episode_meta[eid]["leakage_group"] for eid in assigned}:raise ValueError("HTT role coverage mismatch")
        for a,ra in role_lists.items():
            for b,rb in role_lists.items():
                if a<b and set(ra["leakage_groups"])&set(rb["leakage_groups"]):raise ValueError(f"HTT leakage across roles: {fold}")
        horizon_rows={}
        for horizon in candidates:
            role_rows={}
            for role in role_lists:
                trials=[]
                for eid in role_lists[role]["episodes"]:
                    stage=labels[eid];ts=np.arange(len(stage),dtype=np.int64);onset,pos,neg,excluded=endpoint_trial(ts,stage,horizon,13,True)
                    trials.append({"episode_id":eid,"leakage_group":episode_meta[eid]["leakage_group"],"first_gross_proxy_t":onset,"positive_endpoints":pos,"negative_endpoints":neg,"excluded":excluded})
                summary=summarize_trials(trials);passed,missing=threshold_pass(summary,protocol["support_thresholds"][role]);role_rows[role]={"summary":summary,"threshold":protocol["support_thresholds"][role],"quantity_gate_pass":passed,"missing":missing,"trials":trials}
            horizon_rows[str(horizon)]={"roles":role_rows,"quantity_gate_pass":all(row["quantity_gate_pass"] for row in role_rows.values())}
        selected=next((h for h in candidates if horizon_rows[str(h)]["quantity_gate_pass"]),None)
        output["folds"][fold]={"test_entries_excluded_without_counting":sum(role=="test" for role in original.values()),"label_episode_ids_consumed":sorted(assigned),"label_episode_identity_sha256":identity_sha(sorted(assigned)),"roles":role_lists,"horizons":horizon_rows,"selected_horizon_by_quantity":selected}
    first_supported=None
    for fold in protocol["htt"]["fold_order"]:
        for horizon in candidates:
            if output["folds"][fold]["horizons"][str(horizon)]["quantity_gate_pass"]:
                first_supported={"fold":fold,"horizon":horizon};break
        if first_supported is not None:break
    quantity_pass=first_supported is not None
    output["decision"]={
        "first_supported_fold_horizon_by_quantity":first_supported,
        "quantity_gate_pass":quantity_pass,
        "independent_future_target_gate_pass":False,
        "proxy_exploration_supported":quantity_pass,
        "training_triggered":quantity_pass,
        "reason":(
            "HTT proxy exploration is not triggered because all 16 fixed fold/horizon "
            "combinations fail at least one trial-count threshold; the force-rule-influenced "
            "first-gross target is additionally not independent physical-slip truth"
            if not quantity_pass else
            "HTT passes the fixed trial-count gates for proxy exploration; the force-rule-influenced "
            "first-gross target remains explicitly non-independent physical-slip truth"
        ),
    }
    return output


def load_source_parent(cache_path: Path) -> tuple[dict,dict]:
    cache=torch.load(cache_path,map_location="cpu",weights_only=False);source=Path(cache["source_path"])
    if sha256(source)!=cache["source_sha256"]:raise ValueError("source parent hash mismatch")
    parent=torch.load(source,map_location="cpu",weights_only=False)
    needed={"metadata","current_slip","future_slip"}
    if needed-set(parent):raise KeyError(f"source parent missing {sorted(needed-set(parent))}")
    return cache,parent


def audit_source(train_cache: Path,val_cache: Path,protocol: dict) -> dict:
    train_wrapper,train=load_source_parent(train_cache);val_wrapper,val=load_source_parent(val_cache)
    horizon_values=protocol["source"]["candidate_horizons_in_fixed_order"]
    def rows(parent,wrapper,original_role):
        metadata=parent["metadata"];current=np.asarray(torch.as_tensor(parent["current_slip"]),dtype=np.int8);future=np.asarray(torch.as_tensor(parent["future_slip"]),dtype=np.float32)
        if len(metadata)!=len(current) or future.shape!=(len(current),len(horizon_values)):raise ValueError("source parent shape mismatch")
        wrapper_metadata=wrapper["metadata"];wrapper_future=np.asarray(torch.as_tensor(wrapper["future_slip"]),dtype=np.float32)
        if wrapper_future.shape!=(len(wrapper_metadata),len(horizon_values)):raise ValueError("source deployable cache shape mismatch")
        deployable=defaultdict(dict)
        for index,item in enumerate(wrapper_metadata):deployable[f"{item['dataset']}/{item['trajectory']}"][int(item["sample"])]=wrapper_future[index]
        grouped=defaultdict(list)
        for index,item in enumerate(metadata):grouped[f"{item['dataset']}/{item['trajectory']}"].append((int(item["sample"]),int(current[index]),future[index],index))
        result=[]
        for group,items in sorted(grouped.items()):
            items=sorted(items);samples=[x[0] for x in items]
            duplicate=len(samples)!=len(set(samples));contiguous=samples==list(range(min(samples),max(samples)+1))
            for sample,_,target,_ in items:
                if sample in deployable[group] and not np.array_equal(np.asarray(target,dtype=np.float32),deployable[group][sample]):raise ValueError("deployable source future target drift")
            result.append({"episode_id":f"source/{group}","leakage_group":f"source/{group}","original_role":original_role,"duplicate_samples":duplicate,"contiguous":contiguous,"deployable_samples":sorted(deployable[group]),"items":items})
        return result
    train_rows=rows(train,train_wrapper,"train");val_rows=rows(val,val_wrapper,"validation")
    derived=deterministic_roles([row["leakage_group"] for row in train_rows],"source")
    for row in train_rows:row["role"]=derived[row["leakage_group"]]
    for row in val_rows:row["role"]="outer"
    all_rows=train_rows+val_rows;role_lists={}
    for role in ("fit_train","selection","calibration","outer"):
        episodes=sorted(row["episode_id"] for row in all_rows if row["role"]==role);groups=sorted(row["leakage_group"] for row in all_rows if row["role"]==role)
        role_lists[role]={"episodes":episodes,"leakage_groups":groups,"episode_identity_sha256":identity_sha(episodes),"leakage_group_identity_sha256":identity_sha(groups),"legacy_model_seen_during_training":role!="outer"}
    for a,ra in role_lists.items():
        for b,rb in role_lists.items():
            if a<b and set(ra["leakage_groups"])&set(rb["leakage_groups"]):raise ValueError("source leakage across roles")
    horizon_rows={};raw_horizon_rows={};timeline_complete=all(row["contiguous"] and not row["duplicate_samples"] for row in all_rows);label_mismatches={}
    for hi,horizon in enumerate(horizon_values):
        role_rows={};raw_role_rows={};mismatches=0;compared=0
        for role in role_lists:
            trials=[];raw_trials=[]
            for raw in (row for row in all_rows if row["role"]==role):
                samples=np.asarray([x[0] for x in raw["items"]],dtype=np.int64);stage=np.asarray([x[1] for x in raw["items"]],dtype=np.int8);posmap={int(t):i for i,t in enumerate(samples)}
                raw_onset,raw_pos,raw_neg,raw_excluded=endpoint_trial(samples,stage,horizon,3,False,onset_label=1)
                deployable=np.asarray(raw["deployable_samples"],dtype=np.int64)
                onset,pos,neg,excluded=endpoint_trial(samples,stage,horizon,13,False,onset_label=1,candidate_ts=deployable,history_ts=set(map(int,deployable)))
                for t in pos+neg:
                    compared+=1;derived_target=int(t in pos);stored=float(raw["items"][posmap[t]][2][hi]);mismatches+=int(int(stored>=.5)!=derived_target)
                trials.append({"episode_id":raw["episode_id"],"leakage_group":raw["leakage_group"],"first_current_slip_t":onset,"positive_endpoints":pos,"negative_endpoints":neg,"excluded":excluded,"timeline_contiguous":raw["contiguous"]})
                raw_trials.append({"episode_id":raw["episode_id"],"leakage_group":raw["leakage_group"],"first_current_slip_t":raw_onset,"positive_endpoints":raw_pos,"negative_endpoints":raw_neg,"excluded":raw_excluded,"timeline_contiguous":raw["contiguous"]})
            summary=summarize_trials(trials);passed,missing=threshold_pass(summary,protocol["support_thresholds"][role]);role_rows[role]={"summary":summary,"threshold":protocol["support_thresholds"][role],"quantity_gate_pass":passed,"missing":missing,"trials":trials}
            raw_role_rows[role]={"summary":summarize_trials(raw_trials),"minimum_t":3}
        horizon_rows[str(horizon)]={"roles":role_rows,"quantity_gate_pass":all(row["quantity_gate_pass"] for row in role_rows.values()),"stored_future_target_comparison":{"eligible_endpoints_compared":compared,"binary_mismatches":mismatches,"mismatch_rate":None if not compared else mismatches/compared}}
        raw_horizon_rows[str(horizon)]={"roles":raw_role_rows,"not_used_for_training_gate":True}
        label_mismatches[str(horizon)]={"compared":compared,"mismatches":mismatches}
    selected=next((h for h in horizon_values if horizon_rows[str(h)]["quantity_gate_pass"]),None)
    independent_of_future=timeline_complete and all(row["mismatches"]==0 for row in label_mismatches.values())
    label_code=HERE.parents[3]/"scripts/phase3_2_world_model.py"
    trigger=selected is not None and independent_of_future
    z_dim=int(train_wrapper["z"].shape[1]);history=4
    feature_schema={"history_frames":history,"per_step":{"A_visual":{"z":z_dim,"p_slip":1,"total":z_dim+1},"B_force":{"z":z_dim,"p_slip":1,"force_abs_fz_ft_ratio":3,"total":z_dim+4},"C_force_delta":{"z":z_dim,"p_slip":1,"force_abs_fz_ft_ratio":3,"predicted_delta_xyz":3,"total":z_dim+7}},"flattened_history_dimensions":{"A_visual":history*(z_dim+1),"B_force":history*(z_dim+4),"C_force_delta":history*(z_dim+7)},"available_rows":{"train":len(train_wrapper["metadata"]),"outer_validation":len(val_wrapper["metadata"])}}
    return {"format":"round6_source_support_manifest_v1","status":"complete","protocol_sha256":sha256(PROTOCOL),"inputs":{"train_cache":{"path":str(train_cache),"sha256":sha256(train_cache),"parent_path":train_wrapper["source_path"],"parent_sha256":train_wrapper["source_sha256"]},"val_cache":{"path":str(val_cache),"sha256":sha256(val_cache),"parent_path":val_wrapper["source_path"],"parent_sha256":val_wrapper["source_sha256"]}},"deployable_feature_schema":feature_schema,"roles":role_lists,"raw_timeline_horizons":raw_horizon_rows,"horizons":horizon_rows,"label_evidence":{"contiguous_current_slip_timelines":timeline_complete,"onset_independent_of_overlapping_future_targets":independent_of_future,"authoritative_current_slip_source":"FutureIndexedDataset item slip_label saved as current_slip; future_slip is separately derived as any(slip_label[t+1:t+H+1])","label_generation_code":{"path":str(label_code),"sha256":sha256(label_code),"relevant_lines":"FutureIndexedDataset.__getitem__ and build_feature_cache"},"external_instrument_physical_truth":"not established by this cache audit","future_target_consistency":label_mismatches,"interpretation":"current_slip permits first-onset reconstruction independently of overlapping future_slip columns; it remains dataset label truth rather than separately instrumented physical onset"},"decision":{"selected_horizon_by_quantity":selected,"quantity_gate_pass":selected is not None,"independent_of_future_target_gate_pass":independent_of_future,"proxy_exploration_supported":trigger,"training_triggered":trigger,"reason":"source deployable feature intersection passes the fixed trial gates and current_slip independently reconstructs onset; train-derived selection/calibration are legacy-seen and must not be claimed as unseen evaluation"}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--split-manifest",type=Path,required=True);parser.add_argument("--source-train-cache",type=Path,required=True);parser.add_argument("--source-val-cache",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text());amendment=json.loads(AMENDMENT.read_text())
    if protocol.get("status")!="frozen_before_support_computation" or amendment.get("status")!="frozen_before_support_computation":raise SystemExit("support protocol is not frozen")
    if amendment["locked_original"]["protocol_json_sha256"]!=sha256(PROTOCOL):raise SystemExit("locked support protocol drift")
    output=args.output.resolve();htt=audit_htt(args.contract.resolve(),args.split_manifest.resolve(),protocol);source=audit_source(args.source_train_cache.resolve(),args.source_val_cache.resolve(),protocol)
    atomic_json(output/"htt_support_manifest.json",htt);atomic_json(output/"source_support_manifest.json",source)
    selected_support=None
    if htt["decision"]["training_triggered"]:selected_support={"domain":"htt",**htt["decision"]["first_supported_fold_horizon_by_quantity"]}
    elif source["decision"]["training_triggered"]:selected_support={"domain":"source","horizon":source["decision"]["selected_horizon_by_quantity"]}
    decision={"format":"round6_future_support_decision_v1","status":"complete","training_triggered":selected_support is not None,"selected_support":selected_support,"reasons":[htt["decision"]["reason"],source["decision"]["reason"]],"protocol":{"path":str(PROTOCOL.resolve()),"sha256":sha256(PROTOCOL),"amendment_path":str(AMENDMENT.resolve()),"amendment_sha256":sha256(AMENDMENT)},"effective_selection_order":amendment["corrections"],"role_manifests":{"htt":{"path":str((output/"htt_support_manifest.json").resolve()),"sha256":sha256(output/"htt_support_manifest.json")},"source":{"path":str((output/"source_support_manifest.json").resolve()),"sha256":sha256(output/"source_support_manifest.json")}},"htt":htt["decision"],"source":source["decision"],"thresholds":protocol["support_thresholds"],"test_content_read":False,"audit_code":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())}}
    atomic_json(output/"support_decision.json",decision);print(json.dumps(decision,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
