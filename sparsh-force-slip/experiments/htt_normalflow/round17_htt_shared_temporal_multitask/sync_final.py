#!/usr/bin/env python3
"""Sync the frozen compact delivery to both server mirrors."""
import argparse,json,subprocess
from pathlib import Path
import multitask_train as mt
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).parent);p.add_argument("--host",default="zjy-4090");p.add_argument("--server-code",required=True);p.add_argument("--server-delivery",required=True);a=p.parse_args();manifest=json.loads((a.root/"DELIVERY_MANIFEST.json").read_text());assert manifest["status"]=="ready_for_sync"
 for dst in (a.server_code,a.server_delivery):subprocess.run(["rsync","-a","--exclude","__pycache__/",str(a.root)+"/",f"{a.host}:{dst}/"],check=True)
 receipt={"schema":"round17_sync_receipt_v1","status":"synced_pending_readonly_verify","manifest_sha256":mt.sha(a.root/"DELIVERY_MANIFEST.json"),"destinations":[f"{a.host}:{a.server_code}",f"{a.host}:{a.server_delivery}"]};mt.atomic_json(receipt,a.root/"SYNC_RECEIPT.json");print(json.dumps(receipt,indent=2))
 for dst in (a.server_code,a.server_delivery):subprocess.run(["rsync","-a",str(a.root/"SYNC_RECEIPT.json"),f"{a.host}:{dst}/SYNC_RECEIPT.json"],check=True)
if __name__=="__main__":main()
