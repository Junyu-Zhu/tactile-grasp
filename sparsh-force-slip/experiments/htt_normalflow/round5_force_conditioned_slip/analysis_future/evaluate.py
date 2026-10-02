#!/usr/bin/env python3
"""Independent, preregistered evaluation for Round-5 source and HTT future heads."""
from __future__ import annotations

import argparse,csv,hashlib,importlib.util,json,math,os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
FUTURE=HERE.parent/"future"
EVALUATION_PROTOCOL=HERE/"protocol.json"
spec=importlib.util.spec_from_file_location("r5future_frozen",FUTURE/"future_pipeline.py")
r5=importlib.util.module_from_spec(spec);spec.loader.exec_module(r5)
SEEDS=(20260914,20260915,20260916)
OPS=("fixed_0.5","max_ba","fpr_0.01","fpr_0.05","fpr_0.10")


def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def atomic_json(path:Path,payload:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,sort_keys=True)+"\n");os.replace(tmp,path)


def confusion(y,s,threshold):
    y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64);p=s>=threshold
    tp=int(np.sum(p&(y==1)));fp=int(np.sum(p&(y==0)));fn=int(np.sum(~p&(y==1)));tn=int(np.sum(~p&(y==0)))
    recall=tp/max(1,tp+fn);fpr=fp/max(1,fp+tn);specificity=tn/max(1,tn+fp)
    f1p=2*tp/max(1,2*tp+fp+fn);f1n=2*tn/max(1,2*tn+fp+fn)
    return {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"recall":recall,"fpr":fpr,"balanced_accuracy":(recall+specificity)/2,"macro_f1":(f1p+f1n)/2}


def metrics(y,s,threshold):
    y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
    if set(y.tolist())!={0,1}:raise ValueError("metrics require both classes")
    if not np.isfinite(s).all() or np.any((s<0)|(s>1)):raise ValueError("invalid probability")
    return {**confusion(y,s,threshold),"average_precision":float(average_precision_score(y,s)),"brier":float(np.mean((s-y)**2)),"prevalence":float(y.mean()),"frames":len(y)}


def calibrate(y,s):
    candidates=sorted(set(map(float,s)))+[math.nextafter(1.,math.inf)]
    rows=[(t,confusion(y,s,t)) for t in candidates]
    best=max(rows,key=lambda x:(x[1]["balanced_accuracy"],-x[1]["fpr"],x[0]))
    result={"fixed_0.5":.5,"max_ba":best[0]}
    for cap in (.01,.05,.10):
        valid=[x for x in rows if x[1]["fpr"]<=cap+1e-15]
        result[f"fpr_{cap:.2f}"]=max(valid,key=lambda x:(x[1]["recall"],-x[1]["fpr"],x[0]))[0]
    return result


def cluster_bootstrap(rows,score_key,threshold,seed,repetitions=200):
    groups=defaultdict(list)
    for row in rows:groups[row["cluster"]].append(row)
    names=sorted(groups);rng=np.random.default_rng(seed);values=defaultdict(list)
    for _ in range(repetitions):
        sampled=rng.choice(names,len(names),replace=True);rep=[]
        for name in sampled:rep.extend(groups[name])
        y=np.asarray([r["target"] for r in rep]);s=np.asarray([r[score_key] for r in rep])
        if set(y.tolist())!={0,1}:continue
        m=metrics(y,s,threshold)
        for key in ("balanced_accuracy","macro_f1","fpr","recall","average_precision","brier"):values[key].append(m[key])
    return {key:{"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975)),"valid_replicates":len(v)} for key,v in values.items()}


def per_trial_metrics(rows,score_key,threshold):
    output=[]
    for episode,items in _groups(rows,"episode_id").items():
        y=np.asarray([r["target"] for r in items]);s=np.asarray([r[score_key] for r in items]);c=confusion(y,s,threshold)
        output.append({"episode_id":episode,"cluster":items[0]["cluster"],"frames":len(items),"positive_frames":int(y.sum()),"mean_score":float(s.mean()),"tn":c["tn"],"fp":c["fp"],"fn":c["fn"],"tp":c["tp"]})
    return output


def verified_run(run:Path,domain:str,variant:str,seed:int,protocol_sha:str,allow_smoke=False,inventory=None):
    summary_path=run/"summary.json";summary=json.loads(summary_path.read_text())
    if summary.get("status")!="complete" or summary.get("variant")!=variant or int(summary.get("seed",-1))!=seed:raise ValueError("run identity/status mismatch")
    if not allow_smoke and (summary.get("smoke") is True or summary.get("formal") is not True):raise ValueError("formal evaluation refuses smoke/nonformal run")
    cfg=summary["run_config"]
    if cfg.get("domain")!=domain or cfg.get("protocol_sha256")!=protocol_sha:raise ValueError("run protocol/domain mismatch")
    checkpoints={}
    for name in ("best","latest"):
        item=summary["artifacts"][name];p=Path(item["path"])
        if sha256(p)!=item["sha256"]:raise ValueError("checkpoint hash mismatch")
        checkpoints[name]=torch.load(p,map_location="cpu",weights_only=False)
    for checkpoint in checkpoints.values():
        if checkpoint.get("run_identity_sha256")!=summary.get("run_identity_sha256") or checkpoint.get("run_config")!=cfg:raise ValueError("checkpoint/summary identity mismatch")
    if cfg.get("code_bundle_sha256")!=r5.code_bundle_sha256():raise ValueError("frozen future code-bundle mismatch")
    receipt_path=summary_path.with_name(summary_path.name+".scheduler_receipt.json")
    if not allow_smoke:
        if not receipt_path.is_file():raise ValueError("formal run lacks scheduler receipt")
        receipt=json.loads(receipt_path.read_text());argv=receipt.get("argv",[])
        matches=[] if inventory is None else [job for job in inventory.get("jobs",[]) if Path(job["acceptance_path"]).resolve()==summary_path.resolve()]
        if len(matches)!=1 or receipt.get("argv")!=matches[0].get("argv"):raise ValueError("scheduler receipt is not bound to exactly one frozen inventory job")
        expected_cmd="train-source" if domain=="source" else "train-htt"
        def option(flag):
            if flag not in argv:raise ValueError(f"scheduler receipt lacks {flag}")
            return argv[argv.index(flag)+1]
        if expected_cmd not in argv or option("--variant")!=variant or int(option("--seed"))!=seed or Path(option("--output")).resolve()!=run.resolve():raise ValueError("scheduler receipt job identity mismatch")
        for raw,digest in receipt.get("input_sha256",{}).items():
            path=Path(raw)
            if not path.is_file() or sha256(path)!=digest:raise ValueError("scheduler receipt input drift")
    summary["_verified_provenance"]={"summary_path":str(summary_path.resolve()),"summary_sha256":sha256(summary_path),"scheduler_receipt_path":str(receipt_path.resolve()) if receipt_path.exists() else None,"scheduler_receipt_sha256":sha256(receipt_path) if receipt_path.exists() else None}
    return summary,checkpoints["best"]


def score_diagnostics(scores,threshold):
    scores=np.asarray(scores,dtype=np.float64);decisions=np.unique(scores>=threshold).tolist()
    return {"minimum":float(scores.min()),"maximum":float(scores.max()),"mean":float(scores.mean()),"std":float(scores.std()),"unique_scores":int(len(np.unique(scores))),"decision_classes":[bool(x) for x in decisions],"single_class_decisions":len(decisions)==1}


def position_scores(train_rows,target_rows):
    max_t=max(r["t"] for r in train_rows)
    feat=lambda rows:np.asarray([[min(r["t"]/max_t,1.),min(r["t"]/max_t,1.)**2] for r in rows])
    y=np.asarray([r["target"] for r in train_rows])
    if set(y.tolist())!={0,1}:raise ValueError("position baseline train lacks both classes")
    model=LogisticRegression(C=1.,class_weight="balanced",solver="lbfgs",random_state=20260915,max_iter=1000).fit(feat(train_rows),y)
    return model.predict_proba(feat(target_rows))[:,1]


def exact_source_history(cache):
    source_path=Path(cache["source_path"])
    if sha256(source_path)!=cache["source_sha256"]:raise ValueError("source history parent hash mismatch")
    source=torch.load(source_path,map_location="cpu",weights_only=False);lookup={}
    for i,m in enumerate(source["metadata"]):
        key=(str(m["dataset"]),str(m["trajectory"]),int(m["sample"]))
        if key in lookup:raise ValueError("duplicate source history frame")
        lookup[key]=float(source["slip_probs"][i,1])
    result={}
    for i,m in enumerate(cache["metadata"]):
        key=(str(m["dataset"]),str(m["trajectory"]));t=int(m["sample"]);values=[lookup.get((*key,t-j)) for j in range(3,-1,-1)]
        if all(v is not None for v in values):result[i]=float(np.mean(values))
    return result


def source_rows(cache,probs,history):
    rows=[]
    for i,m in enumerate(cache["metadata"]):
        cluster=f"{m['dataset']}::{m['trajectory']}"
        for hidx,h in enumerate(cache["horizons"]):
            row={"episode_id":cluster,"cluster":cluster,"t":int(m["sample"]),"horizon":int(h),"target":int(cache["future_slip"][i,hidx]),"raw":float(probs[i,hidx]),"slip_only":float(cache["p_slip"][i])}
            if i in history:row["slip_history4_mean"]=history[i]
            rows.append(row)
    return rows


def infer_source(cache_path,checkpoint,variant):
    cache=torch.load(cache_path,map_location="cpu",weights_only=False);x=r5.source_input(cache,variant)
    model=r5.LegacyFutureMLP(x.shape[1],checkpoint["run_config"]["hidden_dim"],len(cache["horizons"]),checkpoint["run_config"]["dropout"])
    model.load_state_dict(checkpoint["model_state"]);model.eval()
    with torch.inference_mode():probs=1-torch.sigmoid(model(x)).numpy()
    return cache,probs


def evaluate_source(runs_root,train_cache,val_cache,future_protocol,output,allow_smoke=False,inventory_path=None):
    psha=sha256(future_protocol);tr=torch.load(train_cache,map_location="cpu",weights_only=False);va=torch.load(val_cache,map_location="cpu",weights_only=False)
    if tr["protocol_sha256"]!=psha or va["protocol_sha256"]!=psha:raise ValueError("source cache protocol mismatch")
    inventory=None if inventory_path is None else json.loads(inventory_path.read_text());train_history=exact_source_history(tr);val_history=exact_source_history(va);results=[];trial_rows=[]
    for variant in r5.SOURCE_VARIANTS:
        for seed in SEEDS:
            summary,ckpt=verified_run(runs_root/variant/f"seed_{seed}","source",variant,seed,psha,allow_smoke,inventory)
            cfg=summary["run_config"]
            if cfg.get("train_cache_sha256")!=sha256(train_cache) or cfg.get("val_cache_sha256")!=sha256(val_cache):raise ValueError("source run/cache provenance mismatch")
            ctr,ptr=infer_source(train_cache,ckpt,variant);cva,pva=infer_source(val_cache,ckpt,variant)
            train_rows=source_rows(ctr,ptr,train_history);val_rows=source_rows(cva,pva,val_history)
            for h in ctr["horizons"]:
                a=[r for r in train_rows if r["horizon"]==h];b=[r for r in val_rows if r["horizon"]==h]
                pos=position_scores(a,b)
                for row,p in zip(b,pos):row["position_logistic"]=float(p);row["raw_mul_pslip_fixed"]=row["raw"]*row["slip_only"]
                methods=("raw","raw_mul_pslip_fixed")
                if variant==r5.SOURCE_VARIANTS[0] and seed==SEEDS[0]:methods += ("slip_only","slip_history4_mean","position_logistic")
                for method in methods:
                    cohort=[r for r in b if method in r];y=[r["target"] for r in cohort];s=[r[method] for r in cohort];m=metrics(y,s,.5)
                    ci=cluster_bootstrap(cohort,method,.5,20260915+seed+int(h));per_trial=per_trial_metrics(cohort,method,.5)
                    coverage={"evaluated_frames":len(cohort),"available_fraction":len(cohort)/len(b),"cohort":"exact_t-3_to_t_history_subset" if method=="slip_history4_mean" else "full_common_cohort"}
                    results.append({"variant":variant if method.startswith("raw") else "shared_baseline","seed":seed if method.startswith("raw") else None,"horizon":int(h),"method":method,"operating_point":"fixed_0.5","threshold":.5,"validation_observed_no_alarm":m["tp"]+m["fp"]==0,"validation_score_diagnostics":score_diagnostics(s,.5),"coverage":coverage,"validation":m,"cluster_bootstrap_95ci":ci,"per_trial":per_trial,"run_provenance":summary["_verified_provenance"] if method.startswith("raw") else None})
                    trial_rows.extend({"domain":"source","variant":variant if method.startswith("raw") else "shared_baseline","seed":seed if method.startswith("raw") else None,"horizon":int(h),"method":method,**row} for row in per_trial)
    provenance_paths=[train_cache,val_cache,future_protocol,EVALUATION_PROTOCOL,Path(__file__)]
    if inventory_path is not None:provenance_paths.append(inventory_path)
    payload={"status":"complete","formal":not allow_smoke,"domain":"source","evaluation_protocol_sha256":sha256(EVALUATION_PROTOCOL),"calibration":"none; fixed 0.5 only","event_metrics":"not reported: future-any-slip frame labels do not provide an independently verified event-onset annotation","results":results,"trial_rows":trial_rows,"provenance":_prov(provenance_paths)}
    _write(output,payload,trial_rows);return payload


def _groups(rows,key):
    out=defaultdict(list)
    for r in rows:out[r[key]].append(r)
    return out


def htt_examples(payload,horizon,role=None):
    examples=[];epsilon=float(payload["condition_normalization"]["ratio_epsilon_n"])
    for raw in payload["episodes"]:
        ep=r5._episode_arrays(raw)
        if role is not None and ep["role"]!=role:continue
        pos={int(t):i for i,t in enumerate(ep["t"].tolist())};gross=[t for t in pos if int(ep["stage"][pos[t]])==2];onset=min(gross) if gross else None
        for t in ep["t"].tolist():
            t=int(t)
            if t<13 or any(any((t-j-lag) not in pos for lag in (0,5,10)) for j in range(4)):continue
            seq=[]
            for cur in range(t-3,t+1):
                i,j=pos[cur],pos[cur-5];z=ep["z"][i].flatten();ps=ep["p_slip"][i].reshape(1);f,fp=ep["force_pred_n"][i],ep["force_pred_n"][j]
                fn,ft=f[2].abs(),torch.linalg.vector_norm(f[:2]);fnp,ftp=fp[2].abs(),torch.linalg.vector_norm(fp[:2])
                seq.append({"risk":torch.cat((z,ps)),"base":torch.cat((z,ps,torch.stack((fn,ft,ft/(fn+epsilon),fn-fnp,ft-ftp))))})
            eligible=(int(ep["stage"][pos[t]])==0 and (onset is None or t<onset) and all((t+j) in pos for j in range(1,horizon+1)))
            target=int(bool(eligible and onset is not None and t<onset<=t+horizon)) if eligible else None
            state=None
            if eligible:
                z0=ep["z"][pos[t]].flatten();state=torch.cat([ep["z"][pos[t+d]].flatten()-z0 for d in (1,3,horizon)])
            examples.append({"episode_id":ep["episode_id"],"role":ep["role"],"t":t,"stage":int(ep["stage"][pos[t]]),"onset":onset,"eligible":eligible,"target":target,"risk_x":torch.stack([x["risk"] for x in seq]),"base_x":torch.stack([x["base"] for x in seq]),"p_slip":float(ep["p_slip"][pos[t]]),"p_history":float(ep["p_slip"][[pos[t-j] for j in range(3,-1,-1)]].mean()),"state_target":state})
    return examples


def infer_htt(examples,checkpoint,variant):
    key="risk_x" if variant=="risk" else "base_x";x=torch.stack([r[key] for r in examples]);norm=checkpoint["normalization"];x=(x-norm["mean"])/norm["std"]
    outdim=int(checkpoint["run_config"]["state_output_dim"]);model=r5.GRURisk(x.shape[2],outdim,variant=="full_state",checkpoint["run_config"]["hidden_dim"]);model.load_state_dict(checkpoint["model_state"]);model.eval()
    with torch.inference_mode():logit,state=model(x)
    return torch.sigmoid(logit).numpy(),None if state is None else state.numpy()


def event_metrics(rows,score,threshold,horizon):
    records=[]
    for eid,items in _groups(rows,"episode_id").items():
        onset=items[0]["onset"]
        if onset is None:continue
        eligible_positive=[r for r in items if r["eligible"] and r["target"]==1]
        if not eligible_positive:
            records.append({"episode_id":eid,"onset":onset,"status":"censored_incomplete_pre_onset_window","first_alarm":None,"lead_frames":None});continue
        early=sorted(r["t"] for r in eligible_positive if r[score]>=threshold)
        late=sorted(r["t"] for r in items if onset<=r["t"]<=onset+horizon and r[score]>=threshold)
        status="early" if early else ("late" if late else "miss")
        records.append({"episode_id":eid,"onset":onset,"status":status,"first_alarm":early[0] if early else (late[0] if late else None),"lead_frames":onset-early[0] if early else None})
    eligible=[r for r in records if r["status"]!="censored_incomplete_pre_onset_window"];leads=[r["lead_frames"] for r in eligible if r["lead_frames"] is not None]
    return {"events_total":len(records),"events_eligible":len(eligible),"events_censored":len(records)-len(eligible),"early":sum(r["status"]=="early" for r in eligible),"late_diagnostic_full_trace":sum(r["status"]=="late" for r in eligible),"miss":sum(r["status"]=="miss" for r in eligible),"early_recall":sum(r["status"]=="early" for r in eligible)/max(1,len(eligible)),"mean_lead_frames":None if not leads else float(np.mean(leads)),"late_scope":"diagnostic inference on causal t>=13 full trace; outside primary current-static eligible frame cohort","records":records}


def event_cluster_bootstrap(event_records,episode_groups,seed,repetitions=200):
    grouped=defaultdict(list)
    for row in event_records:
        if row["status"]!="censored_incomplete_pre_onset_window":grouped[episode_groups[row["episode_id"]]].append(row)
    names=sorted(grouped)
    if not names:return {"early_recall":None,"mean_lead_frames_hits":None,"valid_clusters":0}
    rng=np.random.default_rng(seed);recalls=[];leads=[]
    for _ in range(repetitions):
        rep=[]
        for name in rng.choice(names,len(names),replace=True):rep.extend(grouped[name])
        recalls.append(float(np.mean([r["status"]=="early" for r in rep])))
        current=[r["lead_frames"] for r in rep if r["lead_frames"] is not None]
        if current:leads.append(float(np.mean(current)))
    interval=lambda v:{"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975)),"valid_replicates":len(v)}
    return {"early_recall":interval(recalls),"mean_lead_frames_hits":None if not leads else interval(leads)}


def check_state_normalizer(train_targets,checkpoint,horizon):
    target=torch.stack(train_targets);mean=target.mean(0,keepdim=True);std=target.std(0,keepdim=True).clamp_min(1e-6);saved=checkpoint["normalization"]
    ok_mean=torch.allclose(mean,saved["state_residual_mean"],rtol=1e-5,atol=1e-6);ok_std=torch.allclose(std,saved["state_residual_std"],rtol=1e-5,atol=1e-6)
    d=target.shape[1]//3;blocks=[std[0,i*d:(i+1)*d] for i in range(3)];copied=any(torch.equal(blocks[i],blocks[j]) for i in range(3) for j in range(i+1,3))
    if target.shape[1]%3 or not ok_mean or not ok_std or copied:raise ValueError("state normalizer failed independent 1/3/H variance check")
    return {"status":"pass","output_dim":target.shape[1],"per_horizon_dim":d,"horizons":[1,3,horizon],"mean_matches":ok_mean,"std_matches":ok_std,"copied_horizon_std":copied}


def state_metrics(true,pred,d):
    rows=[]
    for i,h in enumerate((1,3,"H")):
        y=true[:,i*d:(i+1)*d];p=pred[:,i*d:(i+1)*d];den=np.sum((y-y.mean(0))**2);cos=np.sum(y*p,1)/(np.linalg.norm(y,axis=1)*np.linalg.norm(p,axis=1)+1e-12)
        true_variance=float(np.mean(np.var(y,axis=0,ddof=1)));pred_variance=float(np.mean(np.var(p,axis=0,ddof=1)))
        rows.append({"horizon":h,"mse_raw":float(np.mean((p-y)**2)),"r2_raw":float(1-np.sum((p-y)**2)/den) if den>0 else None,"pooled_cosine_similarity":float(np.mean(cos)),"true_variance":true_variance,"predicted_variance":pred_variance,"variance_replication_ratio":pred_variance/true_variance if true_variance>0 else None})
    return rows


def state_cluster_bootstrap(true,pred,clusters,d,seed,repetitions=200):
    by=defaultdict(list)
    for i,name in enumerate(clusters):by[name].append(i)
    names=sorted(by);rng=np.random.default_rng(seed);values={(h,k):[] for h in range(3) for k in ("mse_raw","r2_raw","variance_replication_ratio")}
    for _ in range(repetitions):
        idx=[]
        for name in rng.choice(names,len(names),replace=True):idx.extend(by[name])
        for hi,row in enumerate(state_metrics(true[idx],pred[idx],d)):
            for key in ("mse_raw","r2_raw","variance_replication_ratio"):
                if row[key] is not None and math.isfinite(row[key]):values[(hi,key)].append(row[key])
    return [{"horizon":(1,3,"H")[hi],**{key:{"lower":float(np.quantile(values[(hi,key)],.025)),"upper":float(np.quantile(values[(hi,key)],.975)),"valid_replicates":len(values[(hi,key)])} if values[(hi,key)] else None for key in ("mse_raw","r2_raw","variance_replication_ratio")}} for hi in range(3)]


def evaluate_htt(runs_root,upstream,support,split_manifest,future_protocol,output,allow_smoke=False,inventory_path=None):
    psha=sha256(future_protocol);payload=r5.canonical_htt_payload(upstream,require_formal=not allow_smoke);audit=json.loads(support.read_text());h=int(audit["selected_horizon"])
    if audit["protocol_sha256"]!=psha:raise ValueError("support protocol mismatch")
    inventory=None if inventory_path is None else json.loads(inventory_path.read_text());manifest=json.loads(split_manifest.read_text());groups={r["id"]:r["leakage_group"] for r in manifest["episodes"]}
    examples=htt_examples(payload,h);train=[r for r in examples if r["role"]=="train" and r["eligible"]];cal=[r for r in examples if r["role"]=="calibration" and r["eligible"]];val=[r for r in examples if r["role"]=="validation" and r["eligible"]]
    if any(r["role"]=="test" for r in examples):raise ValueError("HTT upstream contains test role")
    episode_meta={r["id"]:r for r in manifest["episodes"]};expected={eid:role for role in ("train","calibration","validation") for eid in manifest["splits"]["htt_leave_p1"][role] if episode_meta[eid]["task"]=="slip"}
    observed={r["episode_id"]:r["role"] for r in examples}
    if observed!=expected:raise ValueError("HTT upstream episode roles differ from frozen schema-2 split")
    for r in examples:r["cluster"]=groups[r["episode_id"]]
    role_groups={role:{r["cluster"] for r in examples if r["role"]==role} for role in ("train","calibration","validation")}
    if any(role_groups[a]&role_groups[b] for a,b in (("train","calibration"),("train","validation"),("calibration","validation"))):raise ValueError("HTT role leakage-group overlap")
    position_all=position_scores(train,examples)
    for r,p in zip(examples,position_all):r["position_logistic"]=float(p);r["slip_only"]=r["p_slip"];r["slip_history4_mean"]=r["p_history"]
    results=[];trials=[];state_results=[]
    for variant in r5.HTT_VARIANTS:
        for seed in SEEDS:
            summary,ckpt=verified_run(runs_root/variant/f"seed_{seed}","htt",variant,seed,psha,allow_smoke,inventory)
            cfg=summary["run_config"]
            if cfg.get("upstream_sha256")!=sha256(upstream) or cfg.get("support_audit_sha256")!=sha256(support) or int(cfg.get("horizon",-1))!=h:raise ValueError("HTT run/upstream/support provenance mismatch")
            all_probs,state_pred=infer_htt(examples,ckpt,variant);lookup={(r["episode_id"],r["t"]):float(p) for r,p in zip(examples,all_probs)}
            for r in examples:r[f"{variant}_{seed}_raw"]=lookup[(r["episode_id"],r["t"])];r[f"{variant}_{seed}_mul"]=r[f"{variant}_{seed}_raw"]*r["p_slip"]
            methods=[("raw",f"{variant}_{seed}_raw"),("raw_mul_pslip_fixed",f"{variant}_{seed}_mul")]
            if variant=="risk" and seed==SEEDS[0]:methods += [(x,x) for x in ("slip_only","slip_history4_mean","position_logistic")]
            for method,key in methods:
                thresholds=calibrate([r["target"] for r in cal],[r[key] for r in cal])
                for op,t in thresholds.items():
                    cm=metrics([r["target"] for r in cal],[r[key] for r in cal],t);vm=metrics([r["target"] for r in val],[r[key] for r in val],t);ci=cluster_bootstrap(val,key,t,20260915+seed+sum(map(ord,method+op)))
                    trace=[r for r in examples if r["role"]=="validation"];events=event_metrics(trace,key,t,h);event_ci=event_cluster_bootstrap(events["records"],groups,20260915+seed+sum(map(ord,"event"+method+op)))
                    per_trial=per_trial_metrics(val,key,t);trials.extend({"domain":"htt","variant":variant if method.startswith("raw") else "shared_baseline","seed":seed if method.startswith("raw") else None,"method":method,"operating_point":op,**row} for row in per_trial)
                    results.append({"variant":variant if method.startswith("raw") else "shared_baseline","seed":seed if method.startswith("raw") else None,"method":method,"operating_point":op,"threshold":t,"explicit_never_alarm":t>1,"calibration_observed_no_alarm":cm["tp"]+cm["fp"]==0,"validation_observed_no_alarm":vm["tp"]+vm["fp"]==0,"calibration_score_diagnostics":score_diagnostics([r[key] for r in cal],t),"validation_score_diagnostics":score_diagnostics([r[key] for r in val],t),"calibration":cm,"validation":vm,"validation_cluster_bootstrap_95ci":ci,"validation_events":events,"validation_event_cluster_bootstrap_95ci":event_ci,"per_trial":per_trial,"run_provenance":summary["_verified_provenance"] if method.startswith("raw") else None})
            if variant=="full_state":
                eligible_idx=[i for i,r in enumerate(examples) if r["role"]=="validation" and r["eligible"]];tr_idx=[i for i,r in enumerate(examples) if r["role"]=="train" and r["eligible"]]
                check=check_state_normalizer([examples[i]["state_target"] for i in tr_idx],ckpt,h);norm=ckpt["normalization"];learned=state_pred[eligible_idx]*norm["state_residual_std"].numpy()+norm["state_residual_mean"].numpy();truth=torch.stack([examples[i]["state_target"] for i in eligible_idx]).numpy();d=truth.shape[1]//3
                clusters=[groups[examples[i]["episode_id"]] for i in eligible_idx]
                row={"seed":seed,"normalizer_variance_replication_check":check,"learned_full_state":state_metrics(truth,learned,d),"learned_full_state_cluster_bootstrap_95ci":state_cluster_bootstrap(truth,learned,clusters,d,20260915+seed),"train_samples":len(tr_idx),"validation_samples":len(eligible_idx),"same_eligible_frame_identity_sha256":hashlib.sha256(json.dumps([(examples[i]["episode_id"],examples[i]["t"]) for i in eligible_idx]).encode()).hexdigest()}
                if seed==SEEDS[0]:
                    xtr=np.stack([examples[i]["base_x"].numpy().reshape(-1) for i in tr_idx]);ytr=np.stack([examples[i]["state_target"].numpy() for i in tr_idx]);xv=np.stack([examples[i]["base_x"].numpy().reshape(-1) for i in eligible_idx]);ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(xtr,ytr);linear=ridge.predict(xv)
                    row.update({"persistence_zero_residual":state_metrics(truth,np.zeros_like(truth),d),"persistence_cluster_bootstrap_95ci":state_cluster_bootstrap(truth,np.zeros_like(truth),clusters,d,20260915),"train_only_ridge":state_metrics(truth,linear,d),"train_only_ridge_cluster_bootstrap_95ci":state_cluster_bootstrap(truth,linear,clusters,d,20260916)})
                state_results.append(row)
    provenance_paths=[upstream,support,split_manifest,future_protocol,EVALUATION_PROTOCOL,Path(__file__)]
    if inventory_path is not None:provenance_paths.append(inventory_path)
    payload_out={"status":"complete","formal":not allow_smoke,"domain":"htt","horizon":h,"evaluation_protocol_sha256":sha256(EVALUATION_PROTOCOL),"results":results,"state_evaluation":state_results,"trial_rows":trials,"provenance":_prov(provenance_paths)}
    _write(output,payload_out,trials);return payload_out


def _prov(paths):return {str(Path(p).resolve()):sha256(Path(p).resolve()) for p in paths}


def seed_aggregates(results):
    groups=defaultdict(list)
    for row in results:
        if row.get("seed") is not None:groups[(row["variant"],row["method"],row.get("horizon"),row["operating_point"])].append(row)
    output=[]
    for key,rows in sorted(groups.items(),key=str):
        if len(rows)!=3:continue
        values={}
        for metric_name in ("balanced_accuracy","macro_f1","fpr","recall","average_precision","brier"):
            x=[r["validation"][metric_name] for r in rows];values[metric_name]={"mean":float(np.mean(x)),"std":float(np.std(x,ddof=1))}
        output.append({"variant":key[0],"method":key[1],"horizon":key[2],"operating_point":key[3],"n_seeds":3,"descriptive_not_independent_ci":True,"validation":values})
    return output


def render_report(payload):
    lines=[f"# Round-5 future {payload['domain']} 评价","",f"- formal: `{payload['formal']}`",f"- evaluation protocol SHA256: `{payload['evaluation_protocol_sha256']}`","","所有数值均来自固定协议；共享数据上的 seed 标准差只描述初始化波动。"]
    if payload["domain"]=="source":lines += ["","Source 没有独立 calibration 角色，因此只报告固定 0.5 与阈值无关指标，不提供调阈值后的部署结论。future-any-slip 帧标签不足以验证独立事件 onset，因此不报告 source 事件提前量。"]
    else:lines += ["",f"HTT 任务为当前 static 端点预测 `(t,t+{payload['horizon']}]` 内首次 gross。晚报只是在因果 `t>=13` 全轨迹上的诊断；不属于主 static 资格帧指标。没有完整前窗的 onset 记作 censored。"]
    selected=[r for r in payload["results"] if r["operating_point"] in (("fixed_0.5",) if payload["domain"]=="source" else ("max_ba",))]
    lines += ["","| variant | seed | method | H | BA | FPR | recall | AP | Brier |","|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        v=r["validation"];lines.append(f"| {r['variant']} | {r.get('seed') or '-'} | {r['method']} | {r.get('horizon',payload.get('horizon','-'))} | {v['balanced_accuracy']:.4f} | {v['fpr']:.4f} | {v['recall']:.4f} | {v['average_precision']:.4f} | {v['brier']:.4f} |")
    lines += ["","每个 operating point 的 JSON 同时记录 calibration/validation 是否实际零告警，以及概率标准差、唯一值数和决策类别数。单类决策是需要明确报告的模型结果，不能静默计作成功。","","旧 GT force-delta future 实验使用了不同输入条件，本轮未将其数值并入部署一致主表；仅可在总报告中作为有明确输入差异的历史参照。","","证据限制：validation 参与 checkpoint 选择，属于开发证据；calibration 的 FPR 约束不保证 validation FPR；乘法门控是固定分数组合，不能证明 raw future 概率已校准。",""]
    return "\n".join(lines)


def _write(output,payload,trials):
    output.mkdir(parents=True,exist_ok=True);csvpath=output/"trials.csv"
    if trials:
        tmp=csvpath.with_name(csvpath.name+f".tmp.{os.getpid()}")
        with tmp.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(trials[0]));w.writeheader();w.writerows(trials)
        os.replace(tmp,csvpath);payload["trial_artifact"]={"path":str(csvpath.resolve()),"sha256":sha256(csvpath)}
    payload["seed_aggregates"]=seed_aggregates(payload["results"]);atomic_json(output/"evaluation.json",payload)
    report=output/"REPORT_ZH.md";tmp=report.with_name(report.name+f".tmp.{os.getpid()}");tmp.write_text(render_report(payload));os.replace(tmp,report)


def main():
    evaluation_protocol=json.loads(EVALUATION_PROTOCOL.read_text())
    if evaluation_protocol.get("status")!="frozen_before_formal_future_results":raise SystemExit("evaluation protocol is not frozen")
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("source");a.add_argument("--runs-root",type=Path,required=True);a.add_argument("--train-cache",type=Path,required=True);a.add_argument("--val-cache",type=Path,required=True)
    b=sub.add_parser("htt");b.add_argument("--runs-root",type=Path,required=True);b.add_argument("--upstream",type=Path,required=True);b.add_argument("--support",type=Path,required=True);b.add_argument("--split-manifest",type=Path,required=True)
    for q in (a,b):q.add_argument("--future-protocol",type=Path,required=True);q.add_argument("--inventory",type=Path);q.add_argument("--output",type=Path,required=True);q.add_argument("--allow-smoke",action="store_true")
    x=p.parse_args()
    if not x.allow_smoke and x.inventory is None:raise SystemExit("formal evaluation requires --inventory")
    result=evaluate_source(x.runs_root,x.train_cache,x.val_cache,x.future_protocol,x.output,x.allow_smoke,x.inventory) if x.cmd=="source" else evaluate_htt(x.runs_root,x.upstream,x.support,x.split_manifest,x.future_protocol,x.output,x.allow_smoke,x.inventory)
    print(json.dumps({"status":result["status"],"formal":result["formal"],"results":len(result["results"])}))


if __name__=="__main__":main()
