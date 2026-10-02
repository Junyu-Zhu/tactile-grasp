from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest

import numpy as np

from common import sha256_file
from predict_force import validate_adapt_parent


class ProvenanceTests(unittest.TestCase):
    def test_adapt_parent_closure_and_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source.pth";source.write_bytes(b"source")
            audit=root/"audit.json";audit.write_bytes(b"audit")
            args=argparse.Namespace(source_checkpoint=source,cache_audit=audit,fold="htt_leave_p1",seed=20260914)
            cache={"manifest_sha256":"cache"}
            parent={"format":"round5_htt_native_force_adapter_v1","config":{
                "fold":args.fold,"seed":args.seed,"cache_manifest_sha256":"cache",
                "cache_audit_sha256":sha256_file(audit),"source_checkpoint_sha256":sha256_file(source),
                "target":"clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal","smoke":False},
                "provenance":{"force_output_head_reinitialized":True},
                "normalization":{"mean":np.zeros(3,np.float32),"std":np.ones(3,np.float32)}}
            audit_payload={"status":"pass"}
            validate_adapt_parent(parent,args,cache,audit_payload)
            for field in ("cache_manifest_sha256","cache_audit_sha256","source_checkpoint_sha256"):
                altered={**parent,"config":{**parent["config"],field:"wrong"}}
                with self.assertRaisesRegex(RuntimeError,"provenance mismatch"):
                    validate_adapt_parent(altered,args,cache,audit_payload)


if __name__=="__main__":unittest.main()
