from __future__ import annotations
import unittest
import json,tempfile
from pathlib import Path
import build_analysis_manifest as b
import run_analysis as r

class QueueTests(unittest.TestCase):
    def test_fixed_queue_shape_and_unique_acceptance(self):
        payload=b.build();jobs=payload["jobs"]
        self.assertEqual(len(jobs),82)
        self.assertEqual(len({j["id"] for j in jobs}),82)
        self.assertEqual(len({j["acceptance_path"] for j in jobs}),82)
        self.assertEqual(sum(j["id"].startswith("force_") for j in jobs),24)
        self.assertEqual(sum(j["id"].startswith("sensitivity_") for j in jobs),48)
        self.assertFalse(payload["builder_launches_processes"])

    def test_amendment_is_bound_to_current_bundle(self):
        amendment,identity=r.validated_amendment(r.code_bundle()["sha256"])
        self.assertEqual(amendment["old_bundle_sha256"],"5f04a1bc1392890dcd2b6972bd4fcc0fa088bfb52e180c094e00d7ac4e46f68c")
        self.assertIn("benchmark_e2e.py",amendment["output_affecting_changed_files"][0]);self.assertEqual(len(identity["sha256"]),64)

    def test_current_receipt_binds_acceptance_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/"result.json";output.write_text('{"status":"complete"}')
            job={"argv":["python","missing.py"],"acceptance_path":str(output),"identity":{"status":"complete"}}
            bundle=r.code_bundle()["sha256"];receipt=output.with_name(output.name+".analysis_receipt.json")
            receipt.write_text(json.dumps({"manifest_sha256":"m","code_bundle_sha256":bundle,"acceptance_sha256":r.sha(output),"command":r.identity(job)}))
            self.assertTrue(r.accepted(job,"m",bundle));output.write_text('{"status":"changed"}');self.assertFalse(r.accepted(job,"m",bundle))

    def test_lock_precedes_reuse_proof_write(self):
        source=Path(r.__file__).read_text();self.assertLess(source.index("fcntl.flock(lock"),source.index("reuse_proof=write_reuse_proof"))

if __name__=="__main__":unittest.main()
