#!/usr/bin/env python3
"""Fixed representative end-to-end latency for Round-5 deployment paths."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];CURRENT=HERE.parent/"current"
sys.path.insert(0,str(ROOT/"experiments/htt_normalflow"));import adapters  # noqa:E402
sys.path.insert(0,str(ROOT/"scripts"));import phase2_b_multitask as p2  # noqa:E402
sys.path.insert(0,str(CURRENT))
from common import atomic_json, configure_determinism, sha256_file  # noqa:E402
from metrics import apply_clip_standardize  # noqa:E402
from models import ForceAdapter, ForceConditionedSlip, FrozenOldForce, FrozenSlipBranch, physical_condition  # noqa:E402
from sources import load_decoupled_decoder, load_r3_slip_branch  # noqa:E402
from data import load_cache, require_cache_audit  # noqa:E402
sys.path.insert(0,str(HERE.parent/"future"));import future_pipeline as future  # noqa:E402
from analysis_common import protocol, sha256, verify_receipt_inventory  # noqa:E402
from analyze_force import validate_formal_force_parent  # noqa:E402


def validate_slip_parent(checkpoint, summary_path, args, contract):
    checkpoint=checkpoint.resolve();summary_path=summary_path.resolve()
    parent=torch.load(checkpoint,map_location="cpu",weights_only=False);summary=json.loads(summary_path.read_text());config=parent.get("config",{})
    checks={
        "format":parent.get("format")=="round5_force_conditioned_slip_v1",
        "identity":config.get("variant")==args.variant and config.get("fold")==args.fold and int(config.get("seed",-1))==args.seed,
        "formal":config.get("smoke") is False and summary.get("status")=="complete" and summary.get("smoke") is False,
        "summary_best_path":Path(summary.get("best_checkpoint","")).resolve()==checkpoint,
        "summary_best_sha":summary.get("best_checkpoint_sha256")==sha256(checkpoint),
        "summary_config":all(summary.get(k)==v for k,v in config.items()),
        "contract":config.get("cache_manifest_sha256")==contract["manifest_sha256"],
        "audit":config.get("cache_audit_sha256")==sha256(args.cache_audit.resolve()),
        "source":config.get("source_checkpoint_sha256")==sha256(args.source_checkpoint.resolve()),
        "base":config.get("base_slip_checkpoint_sha256")==sha256(args.base_slip_checkpoint.resolve()),
    }
    proof=verify_receipt_inventory(summary_path,args.inventory)
    if not all(checks.values()):raise ValueError(f"formal slip parent provenance mismatch: {[k for k,v in checks.items() if not v]}")
    return parent,{"summary_sha256":sha256(summary_path),**proof}


def validate_future_parent(checkpoint, summary_path, inventory_path, args):
    checkpoint=checkpoint.resolve();summary_path=summary_path.resolve();inventory_path=inventory_path.resolve()
    parent=torch.load(checkpoint,map_location="cpu",weights_only=False);summary=json.loads(summary_path.read_text());config=parent.get("run_config",{})
    receipt=summary_path.with_name(summary_path.name+".scheduler_receipt.json");inventory=json.loads(inventory_path.read_text())
    matching=[job for job in inventory.get("jobs",inventory.get("runs",[])) if Path(job.get("acceptance_path","")).resolve()==summary_path]
    checks={"identity":parent.get("variant")=="full_state" and int(parent.get("seed",-1))==args.seed,
            "formal":summary.get("status")=="complete" and summary.get("formal") is True and summary.get("smoke") is False,
            "best_path":Path(summary.get("artifacts",{}).get("best",{}).get("path","")).resolve()==checkpoint,
            "best_sha":summary.get("artifacts",{}).get("best",{}).get("sha256")==sha256(checkpoint),
            "config":summary.get("run_config")==config and parent.get("run_identity_sha256")==summary.get("run_identity_sha256"),
            "receipt":receipt.is_file(),"inventory_match":len(matching)==1}
    if receipt.is_file():
        payload=json.loads(receipt.read_text());checks["receipt_inputs"]=all(Path(p).is_file() and sha256(Path(p))==digest for p,digest in payload.get("input_sha256",{}).items())
        if matching:checks["receipt_argv"]=payload.get("argv")==matching[0].get("argv")
    if not all(checks.values()):raise ValueError(f"formal future parent provenance mismatch: {[k for k,v in checks.items() if not v]}")
    return parent,{"summary_sha256":sha256(summary_path),"scheduler_receipt_sha256":sha256(receipt),"inventory_sha256":sha256(inventory_path)}


def synchronize(device):
    if str(device).startswith("cuda"):torch.cuda.synchronize(device)


def unique_parameters(modules):
    seen={};trainable=0
    for module in modules:
        for p in module.parameters():
            if id(p) not in seen:seen[id(p)]=p.numel();trainable+=p.numel() if p.requires_grad else 0
    return sum(seen.values()),trainable


class Pipeline:
    def __init__(self,args,entry,episode,contract):
        self.args=args;self.entry=entry;self.episode=episode;self.device=args.device;self.t=13
        decoder,_=load_decoupled_decoder(args.source_checkpoint.resolve())
        torch.manual_seed(42);bundle=p2.FrozenEncoderSharedForceSlip("mae",decoder_variant="decoupled")
        self.encoder=bundle.encoder.eval().requires_grad_(False).to(self.device)
        base=load_r3_slip_branch(decoder,args.base_slip_checkpoint.resolve())
        base_parent=torch.load(args.base_slip_checkpoint.resolve(),map_location="cpu",weights_only=False);base_config=base_parent.get("config",{})
        if base_parent.get("format")!="round3_mae_slip_branch_v1" or base_config.get("init")!="fresh" or base_config.get("fold")!=args.fold or int(base_config.get("seed",-1))!=args.seed or base_config.get("smoke") is not False or base_config.get("source_checkpoint_sha256")!=sha256(args.source_checkpoint.resolve()):
            raise ValueError("benchmark base slip checkpoint is not the matching formal Round3-B parent")
        self.base=FrozenSlipBranch(base).eval().requires_grad_(False).to(self.device)
        self.fusion=None;self.force=None;self.force_norm=None;self.future_model=None;self.future_norm=None
        self.parent_provenance={}
        if args.variant!="mae-b":
            expected="F-adapt" if args.variant=="future-full" else args.variant
            original_variant=args.variant;args.variant=expected
            ck,proof=validate_slip_parent(args.slip_checkpoint,args.slip_training_summary,args,contract)
            args.variant=original_variant;self.parent_provenance["slip"]=proof
            self.fusion=ForceConditionedSlip(base,expected,hidden_dim=64);self.fusion.load_state_dict(ck["model_state"],strict=True);self.fusion.eval().requires_grad_(False).to(self.device)
            self.condition_norm=ck.get("condition_normalization")
        if args.variant in ("F-old","F-adapt","future-full"):
            if args.variant=="F-old":self.force=FrozenOldForce(decoder)
            else:
                fc=validate_formal_force_parent(args.force_checkpoint,args.force_training_summary,contract,args.cache_audit,args.source_checkpoint,args.fold,args.seed,args.inventory)
                self.parent_provenance["force"]={"summary_sha256":sha256(args.force_training_summary.resolve()),"scheduler_receipt_sha256":sha256(args.force_training_summary.with_name(args.force_training_summary.name+".scheduler_receipt.json"))}
                self.force=ForceAdapter(decoder);self.force.load_state_dict(fc["model_state"],strict=True);self.force_norm=fc["normalization"]
            self.force.eval().requires_grad_(False).to(self.device)
        if args.variant=="future-full":
            fc,proof=validate_future_parent(args.future_checkpoint,args.future_summary,args.future_inventory,args);self.parent_provenance["future"]=proof;rc=fc["run_config"]
            if rc.get("variant")!="full_state" or rc.get("execution_mode")!="formal":raise ValueError("wrong/non-formal future checkpoint")
            inp=fc["model_state"]["gru.weight_ih_l0"].shape[1];out=rc["state_output_dim"]
            self.future_model=future.GRURisk(inp,out,True,rc["hidden_dim"]);self.future_model.load_state_dict(fc["model_state"],strict=True);self.future_model.eval().requires_grad_(False).to(self.device)
            self.future_norm=fc["normalization"]
        modules=[self.encoder,(self.base if self.fusion is None else self.fusion)]+([self.force] if self.force else [])+([self.future_model] if self.future_model else [])
        self.parameters=unique_parameters(modules)

    def image(self,t):return adapters.window(self.episode,t)["inputs"]["image"].unsqueeze(0).to(self.device)
    def encode(self,t):return self.encoder(self.image(t))
    def force_value(self,z):
        value=self.force(z)
        if self.force_norm is not None:value=value*torch.as_tensor(self.force_norm["std"],device=self.device)+torch.as_tensor(self.force_norm["mean"],device=self.device)
        return value
    def condition(self,now,previous):
        raw=physical_condition(now,previous,float(self.condition_norm["ratio_epsilon_n"])).cpu().numpy()
        return torch.from_numpy(apply_clip_standardize(raw,self.condition_norm)).to(self.device)
    def detect(self,t,lag_cache=None):
        z=self.encode(t)
        if self.args.variant=="mae-b":return self.base(z)
        if self.args.variant=="V":return self.fusion(z,None)
        now=self.force_value(z);previous=self.force_value(self.encode(t-5)) if lag_cache is None else lag_cache
        return self.fusion(z,self.condition(now,previous))
    def future_input(self,t,stream_cache=None):
        steps=[]
        for cur in range(t-3,t+1):
            if stream_cache is not None and cur<t:
                steps.append(stream_cache[cur]);continue
            z=self.encode(cur);now=self.force_value(z)
            previous=self.force_value(self.encode(cur-5)) if stream_cache is None else stream_cache[(cur,"lag_force")]
            cond=self.condition(now,previous);p=torch.softmax(self.fusion(z,cond),1)[:,1:]
            steps.append(torch.cat((z.mean(1),p,cond),1))
        x=torch.stack(steps,1);mean=self.future_norm["mean"].to(self.device);std=self.future_norm["std"].to(self.device)
        return (x-mean)/std
    def prepare_stream_cache(self):
        if self.args.variant in ("mae-b","V"):return None
        with torch.inference_mode():
            if self.args.variant!="future-full":return self.force_value(self.encode(self.t-5))
            result={}
            for cur in range(self.t-3,self.t):
                z=self.encode(cur);now=self.force_value(z);previous=self.force_value(self.encode(cur-5));cond=self.condition(now,previous);p=torch.softmax(self.fusion(z,cond),1)[:,1:];result[cur]=torch.cat((z.mean(1),p,cond),1)
            result[(self.t,"lag_force")]=self.force_value(self.encode(self.t-5));return result
    def call(self,mode,cache):
        if self.args.variant!="future-full":return self.detect(self.t,cache if mode=="streaming" else None)
        x=self.future_input(self.t,cache if mode=="streaming" else None);return self.future_model(x)[0]


def main():
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,required=True);p.add_argument("--cache-audit",type=Path,required=True);p.add_argument("--inventory",type=Path,required=True);p.add_argument("--source-checkpoint",type=Path,required=True);p.add_argument("--base-slip-checkpoint",type=Path,required=True);p.add_argument("--variant",choices=("mae-b","V","F-old","F-adapt","future-full"),required=True);p.add_argument("--slip-checkpoint",type=Path);p.add_argument("--slip-training-summary",type=Path);p.add_argument("--force-checkpoint",type=Path);p.add_argument("--force-training-summary",type=Path);p.add_argument("--future-checkpoint",type=Path);p.add_argument("--future-summary",type=Path);p.add_argument("--future-inventory",type=Path);p.add_argument("--fold",default="htt_leave_p1");p.add_argument("--seed",type=int,default=20260914);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--warmup",type=int);p.add_argument("--steps",type=int);args=p.parse_args()
    if args.variant!="mae-b" and not all((args.slip_checkpoint,args.slip_training_summary)):p.error("non-mae-b requires --slip-checkpoint and --slip-training-summary")
    if args.variant in ("F-adapt","future-full") and not all((args.force_checkpoint,args.force_training_summary)):p.error("adapted-force path requires --force-checkpoint and --force-training-summary")
    if args.variant=="future-full" and not all((args.future_checkpoint,args.future_summary,args.future_inventory)):p.error("future-full requires --future-checkpoint, --future-summary and --future-inventory")
    configure_determinism();cfg=protocol();contract=load_cache(args.contract.resolve());require_cache_audit(args.cache_audit.resolve(),contract,False);candidates=sorted((e for e in contract["entries"] if e["task"]=="slip" and e["roles_by_fold"][args.fold]=="validation" and e["frames"]>13),key=lambda e:e["episode_id"])
    if not candidates:raise ValueError("no fixed representative validation slip episode")
    entry=candidates[0];source=Path(sorted(entry["source_files"])[0]);
    if sha256(source)!=entry["source_files"][str(source)]:raise ValueError("representative source hash mismatch")
    episode=adapters.load_htt(source);pipe=Pipeline(args,entry,episode,contract);warm=args.warmup or cfg["e2e"]["warmup"];steps=args.steps or cfg["e2e"]["measured_steps"]
    results={}
    for mode in ("cold","streaming"):
        cache=pipe.prepare_stream_cache() if mode=="streaming" else None;times=[]
        if str(args.device).startswith("cuda"):torch.cuda.reset_peak_memory_stats(args.device)
        with torch.inference_mode():
            for i in range(warm+steps):
                synchronize(args.device);start=time.perf_counter();out=pipe.call(mode,cache);synchronize(args.device);elapsed=(time.perf_counter()-start)*1000
                if not torch.isfinite(out).all():raise ValueError("non-finite benchmark output")
                if i>=warm:times.append(elapsed)
        results[mode]={"median_ms":statistics.median(times),"p10_ms":float(np.quantile(times,.1)),"p90_ms":float(np.quantile(times,.9)),"samples_ms":times,"peak_allocated_bytes":torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else 0}
    cold_calls=8 if args.variant=="future-full" else (2 if args.variant in ("F-old","F-adapt") else 1)
    payload={"status":"complete","format":"round5_e2e_latency_v1","variant":args.variant,"fold":args.fold,"seed":args.seed,"representative_selection":"lexicographically first validation slip episode with frame13 available; fixed before results","episode_id":entry["episode_id"],"frame":13,"warmup":warm,"steps":steps,"total_unique_parameters":pipe.parameters[0],"parameters_requiring_grad_at_inference":pipe.parameters[1],"measurement":results,"measurement_semantics":{"resident_input":"raw episode arrays are already resident in memory; NPZ I/O, sensor capture, model construction, checkpoint loading, and historical-cache construction are excluded","cold":f"includes {cold_calls} raw-window preprocessing and encoder calls per measured invocation; future-full reconstructs all four current/lag pairs","streaming":"historical feature/lag-force cache preparation is outside the timed loop; each measured invocation includes raw preprocessing and one current encoder/force/head path","encoder_calls_per_measured_invocation":{"cold":cold_calls,"streaming":1}},"protocol_sha256":sha256(HERE/"protocol.json"),"contract_sha256":sha256(args.contract.resolve()),"cache_audit_sha256":sha256(args.cache_audit.resolve()),"source_checkpoint_sha256":sha256(args.source_checkpoint.resolve()),"base_slip_checkpoint_sha256":sha256(args.base_slip_checkpoint.resolve()),"parent_provenance":pipe.parent_provenance,"source_sha256":sha256(source),"gpu_condition":__import__("subprocess").check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,utilization.gpu","--format=csv"],text=True) if str(args.device).startswith("cuda") else "cpu"}
    atomic_json(args.output.resolve(),payload);print(json.dumps({**payload,"measurement":{k:{x:y for x,y in v.items() if x!="samples_ms"} for k,v in results.items()}},indent=2))


if __name__=="__main__":main()
