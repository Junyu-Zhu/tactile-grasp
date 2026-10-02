#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audit_support import AMENDMENT,PROTOCOL,atomic_json,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--support",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    root=args.support.resolve();decision=json.loads((root/"support_decision.json").read_text());htt=json.loads((root/"htt_support_manifest.json").read_text());source=json.loads((root/"source_support_manifest.json").read_text())
    role_disjoint=True
    for fold in htt["folds"].values():
        groups=[set(row["leakage_groups"]) for row in fold["roles"].values()]
        role_disjoint &= all(not groups[i]&groups[j] for i in range(len(groups)) for j in range(i))
    source_groups=[set(row["leakage_groups"]) for row in source["roles"].values()]
    role_disjoint &= all(not source_groups[i]&source_groups[j] for i in range(len(source_groups)) for j in range(i))
    checks={"status_complete":decision.get("status")==htt.get("status")==source.get("status")=="complete","manifest_hashes":decision["role_manifests"]["htt"]["sha256"]==sha256(root/"htt_support_manifest.json") and decision["role_manifests"]["source"]["sha256"]==sha256(root/"source_support_manifest.json"),"locked_protocol":decision["protocol"]["sha256"]==sha256(PROTOCOL) and decision["protocol"]["amendment_sha256"]==sha256(AMENDMENT),"no_test_content":decision["test_content_read"] is False and htt["label_access"]["always_test_labels_read"] is False,"role_groups_disjoint":role_disjoint,"source_targets_reconstructed_exactly":all(row["stored_future_target_comparison"]["binary_mismatches"]==0 for row in source["horizons"].values()),"trigger_is_source_h1":decision.get("training_triggered") is True and decision.get("selected_support")=={"domain":"source","horizon":1}}
    if not all(checks.values()):raise ValueError(f"support verification failed: {[key for key,value in checks.items() if not value]}")
    result={"format":"round6_future_support_independent_audit_v1","status":"pass","checks":checks,"artifacts":{"decision":sha256(root/"support_decision.json"),"htt":sha256(root/"htt_support_manifest.json"),"source":sha256(root/"source_support_manifest.json")},"audit_code":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())}}
    atomic_json(args.output.resolve(),result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
