#!/usr/bin/env python3
"""Recoverable serial executor for the fixed force-analysis queue."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,subprocess
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def identity(job):
    inputs={}
    for i,value in enumerate(job["argv"]):
        path=Path(value)
        if i and job["argv"][i-1]=="--output":continue
        if path.is_absolute() and path.is_file() and i:inputs[str(path)]=sha(path)
    return {"argv":job["argv"],"input_sha256":inputs}

def code_bundle():
    root=Path(__file__).resolve().parent
    files=sorted(root.glob("*.py"))+[root/"protocol.json"]
    hashes={str(path.resolve()):sha(path.resolve()) for path in files}
    encoded=json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()
    return {"files":hashes,"sha256":hashlib.sha256(encoded).hexdigest()}

def json_digest(payload):return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def verify_declared_artifacts(value):
    if isinstance(value,list):return all(verify_declared_artifacts(item) for item in value)
    if not isinstance(value,dict):return True
    for key,item in value.items():
        if isinstance(key,str) and key.startswith("/") and isinstance(item,str) and len(item)==64:
            path=Path(key)
            if not path.is_file() or sha(path)!=item:return False
        if isinstance(item,str) and item.startswith("/"):
            expected=value.get(key+"_sha256") or (value.get("sha256") if key=="path" else None)
            if expected is not None:
                path=Path(item)
                if not path.is_file() or sha(path)!=expected:return False
        if not verify_declared_artifacts(item):return False
    return True

def validated_amendment(bundle_sha):
    path=Path(__file__).resolve().parent/"BUNDLE_AMENDMENT.json"
    if not path.is_file():return None
    amendment=json.loads(path.read_text());old=amendment.get("old_files",{});new=amendment.get("new_files",{})
    changed={key for key in old if old.get(key)!=new.get(key)};declared=set(amendment.get("changed_files",{}))
    output=set(amendment.get("output_affecting_changed_files",[]));administrative=set(amendment.get("administrative_changed_files",[]))
    checks=(amendment.get("status")=="pass" and set(old)==set(new) and json_digest(old)==amendment.get("old_bundle_sha256")
            and json_digest(new)==amendment.get("new_bundle_sha256")==bundle_sha and changed==declared==output|administrative
            and not output&administrative and all(Path(raw).is_file() and sha(Path(raw))==digest for raw,digest in new.items()))
    if not checks:raise ValueError("invalid bundle amendment")
    return amendment,{"path":str(path),"sha256":sha(path)}

def write_reuse_proof(manifest,manifest_sha,bundle_sha):
    validated=validated_amendment(bundle_sha)
    if validated is None:return None
    amendment,amendment_identity=validated;rows={};output_changed=set(amendment["output_affecting_changed_files"])
    for job in manifest["jobs"]:
        output=Path(job["acceptance_path"]);receipt=output.with_name(output.name+".analysis_receipt.json")
        if not output.is_file() or not receipt.is_file() or str(Path(job["argv"][1]).resolve()) in output_changed:continue
        saved=json.loads(receipt.read_text())
        if saved.get("manifest_sha256")!=manifest_sha or saved.get("code_bundle_sha256")!=amendment["old_bundle_sha256"] or saved.get("command")!=identity(job):continue
        payload=json.loads(output.read_text())
        if any(payload.get(k)!=v for k,v in job["identity"].items()) or not verify_declared_artifacts(payload):continue
        rows[job["id"]]={"acceptance_path":str(output),"acceptance_sha256":sha(output),"old_receipt_path":str(receipt),"old_receipt_sha256":sha(receipt)}
    proof={"format":"round5_analysis_amendment_reuse_proof_v1","status":"pass","manifest_sha256":manifest_sha,"old_bundle_sha256":amendment["old_bundle_sha256"],"new_bundle_sha256":bundle_sha,"amendment":amendment_identity,"verification_timing":"current post-run files reverified during amendment; this does not claim the old receipt recorded acceptance SHA at original completion","outputs":rows}
    path=Path(__file__).resolve().parent/"AMENDMENT_REUSE_PROOF.json";atomic(path,proof);return proof

def amendment_allows(job,manifest_sha,receipt_bundle,bundle_sha):
    validated=validated_amendment(bundle_sha)
    if validated is None:return False
    amendment,amendment_identity=validated
    if amendment.get("old_bundle_sha256")!=receipt_bundle:return False
    entry=Path(job["argv"][1]).resolve() if len(job["argv"])>1 else None
    if str(entry) in set(amendment.get("output_affecting_changed_files",[])):return False
    proof_path=Path(__file__).resolve().parent/"AMENDMENT_REUSE_PROOF.json"
    if not proof_path.is_file():return False
    proof=json.loads(proof_path.read_text());row=proof.get("outputs",{}).get(job["id"],{});output=Path(job["acceptance_path"]);receipt=output.with_name(output.name+".analysis_receipt.json")
    return (proof.get("manifest_sha256")==manifest_sha and proof.get("old_bundle_sha256")==receipt_bundle
            and proof.get("new_bundle_sha256")==bundle_sha and proof.get("amendment")==amendment_identity
            and row.get("acceptance_path")==str(output) and output.is_file() and row.get("acceptance_sha256")==sha(output)
            and row.get("old_receipt_path")==str(receipt) and receipt.is_file() and row.get("old_receipt_sha256")==sha(receipt))

def accepted(job,manifest_sha,bundle_sha):
    path=Path(job["acceptance_path"]);receipt=path.with_name(path.name+".analysis_receipt.json")
    if not path.is_file() or not receipt.is_file():return False
    payload=json.loads(path.read_text())
    if any(payload.get(k)!=v for k,v in job["identity"].items()):return False
    proof=json.loads(receipt.read_text())
    if proof.get("manifest_sha256")!=manifest_sha or proof.get("command")!=identity(job):return False
    receipt_bundle=proof.get("code_bundle_sha256")
    if receipt_bundle==bundle_sha:return proof.get("acceptance_sha256")==sha(path)
    return amendment_allows(job,manifest_sha,receipt_bundle,bundle_sha)

def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f".tmp.{os.getpid()}");temp.write_text(json.dumps(payload,indent=2));os.replace(temp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--execute",action="store_true");args=p.parse_args()
    manifest=json.loads(args.manifest.read_text());digest=sha(args.manifest.resolve());bundle=code_bundle();states={};state_path=args.manifest.with_name("ANALYSIS_STATE.json")
    lock_path=args.manifest.with_name("ANALYSIS_RUNNER.lock");lock=lock_path.open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reuse_proof=write_reuse_proof(manifest,digest,bundle["sha256"]);amendment_identity=(reuse_proof or {}).get("amendment")
    for job in manifest["jobs"]:
        if accepted(job,digest,bundle["sha256"]):
            receipt=json.loads(Path(job["acceptance_path"]).with_name(Path(job["acceptance_path"]).name+".analysis_receipt.json").read_text())
            states[job["id"]]="reused_verified" if receipt.get("code_bundle_sha256")==bundle["sha256"] else "reused_verified_amendment"
            continue
        missing=[x for x in job["ready_paths"] if not Path(x).is_file()]
        if missing:states[job["id"]]="pending_inputs";continue
        if not args.execute:states[job["id"]]="ready_dry_run";continue
        argv=job["argv"]
        env={**os.environ,"XFORMERS_DISABLED":"1"};log_path=Path(job["log_path"]);log_path.parent.mkdir(parents=True,exist_ok=True)
        states[job["id"]]="running";atomic(state_path,{"status":"running","pid":os.getpid(),"active_job":job["id"],"manifest_sha256":digest,"code_bundle":bundle,"amendment":amendment_identity,"states":states})
        with log_path.open("a") as stream:result=subprocess.run(argv,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            states[job["id"]]="failed";atomic(state_path,{"status":"failed","pid":os.getpid(),"active_job":job["id"],"manifest_sha256":digest,"code_bundle":bundle,"amendment":amendment_identity,"states":states});raise RuntimeError(f"analysis job failed: {job['id']}; see {log_path}")
        if not Path(job["acceptance_path"]).is_file():raise RuntimeError(f"analysis job omitted acceptance: {job['id']}")
        acceptance=Path(job["acceptance_path"])
        atomic(acceptance.with_name(acceptance.name+".analysis_receipt.json"),{"manifest_sha256":digest,"code_bundle_sha256":bundle["sha256"],"amendment_sha256":(amendment_identity or {}).get("sha256"),"acceptance_sha256":sha(acceptance),"command":identity(job)})
        if not accepted(job,digest,bundle["sha256"]):raise RuntimeError(f"analysis acceptance failed: {job['id']}")
        states[job["id"]]="complete";atomic(state_path,{"status":"running","pid":os.getpid(),"active_job":None,"manifest_sha256":digest,"code_bundle":bundle,"amendment":amendment_identity,"states":states})
    pending=any(state=="pending_inputs" for state in states.values());status="dry_run_no_process_started" if not args.execute else ("pending" if pending else "pass")
    payload={"status":status,"pid":os.getpid(),"active_job":None,"manifest_sha256":digest,"code_bundle":bundle,"amendment":amendment_identity,"counts":{state:list(states.values()).count(state) for state in sorted(set(states.values()))},"states":states}
    atomic(state_path,payload);print(json.dumps(payload,indent=2))

if __name__=="__main__":main()
