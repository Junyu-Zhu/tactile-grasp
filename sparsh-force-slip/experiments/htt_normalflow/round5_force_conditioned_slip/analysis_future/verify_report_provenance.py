#!/usr/bin/env python3
"""Verify computation-source provenance and report-only rendering for Round 5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from render_reports import render_report


EXPECTED_EVALUATOR_SHA256="433cbbaf6de68115e862ab9276151f8b986582594a71a267be51fb44f85ea8f2"


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def verify(evaluation_path: Path,report_path: Path) -> dict:
    payload=json.loads(evaluation_path.read_text())
    recorded=[value for path,value in payload.get("provenance",{}).items() if Path(path).name=="evaluate.py"]
    expected_report=render_report(payload).encode()
    checks={
        "evaluation_complete":payload.get("status")=="complete",
        "one_recorded_evaluator":recorded==[EXPECTED_EVALUATOR_SHA256],
        "report_matches_renderer":report_path.read_bytes()==expected_report,
    }
    if not all(checks.values()):raise ValueError(f"report provenance failed: {[key for key,value in checks.items() if not value]}")
    return {"domain":payload.get("domain"),"evaluation_path":str(evaluation_path),"evaluation_sha256":sha256(evaluation_path),"recorded_evaluator_sha256":recorded[0],"report_path":str(report_path),"report_sha256":sha256(report_path),"checks":checks}


def main():
    parser=argparse.ArgumentParser()
    for domain in ("source","htt"):
        parser.add_argument(f"--{domain}-evaluation",type=Path,required=True)
        parser.add_argument(f"--{domain}-report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    here=Path(__file__).resolve().parent;evaluator=here/"evaluate.py";renderer=here/"render_reports.py"
    if sha256(evaluator)!=EXPECTED_EVALUATOR_SHA256:raise ValueError("immutable computation evaluator SHA mismatch")
    rows=[verify(getattr(args,f"{domain}_evaluation").resolve(),getattr(args,f"{domain}_report").resolve()) for domain in ("source","htt")]
    result={"format":"round5_future_report_render_provenance_v1","status":"pass","scope":"report-only clarification; evaluation JSON and trial CSV are unchanged","computation_evaluator":{"path":str(evaluator),"sha256":sha256(evaluator)},"report_renderer":{"path":str(renderer),"sha256":sha256(renderer)},"verifier":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())},"artifacts":rows}
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+f".tmp.{os.getpid()}");temporary.write_text(json.dumps(result,indent=2));os.replace(temporary,output)


if __name__=="__main__":main()
