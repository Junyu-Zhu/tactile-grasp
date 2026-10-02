from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import sha256_file
from data import FORCE_TARGET_SEMANTICS, ForceFrames, SlipFrames, load_cache, stratified_smoke_indices


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        roles = {f"htt_leave_p{i}": "train" for i in range(1, 5)}
        entries=[]
        for task in ("force", "slip"):
            token=root/f"{task}.tokens.npy"; np.save(token,np.zeros((20,3,4),np.float32))
            entry={"episode_id":f"htt/{task}","task":task,"roles_by_fold":roles,"frames":20,
                   "token_path":str(token),"token_sha256":sha256_file(token)}
            if task=="force":
                target=root/"force.npy"; np.save(target,np.ones((20,3),np.float32))
                entry.update(force_native_n_path=str(target),force_native_n_sha256=sha256_file(target))
            else:
                target=root/"labels.npy"; np.save(target,np.array([0,1,2,0]*5,np.int64))
                entry.update(label_path=str(target),label_sha256=sha256_file(target))
            entries.append(entry)
        self.manifest=root/"cache_manifest.json"
        self.manifest.write_text(json.dumps({"format":"round5_htt_mae_tokens_targets_v1","status":"complete",
            "force_target_semantics":FORCE_TARGET_SEMANTICS,
            "pairing_policy":"independent_episodes_no_force_slip_basename_join","entries":entries}))

    def tearDown(self): self.temp.cleanup()

    def test_strict_force_and_slip_starts(self):
        cache=load_cache(self.manifest,verify_hashes=True)
        force=[entry for entry in cache["entries"] if entry["task"]=="force"]
        slip=[entry for entry in cache["entries"] if entry["task"]=="slip"]
        force_data=ForceFrames(force); slip_data=SlipFrames(slip)
        self.assertEqual(force_data.index[0][1],5)
        self.assertTrue(all(frame>=10 for _,frame,_,_ in slip_data.index))

    def test_wrong_semantics_refused(self):
        payload=json.loads(self.manifest.read_text());payload["force_target_semantics"]="raw"
        self.manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError,"semantics"):
            load_cache(self.manifest)

    def test_smoke_subset_is_stratified_across_episodes(self):
        cache=load_cache(self.manifest)
        slip=[entry for entry in cache["entries"] if entry["task"]=="slip"]
        second={**slip[0],"episode_id":"htt/slip2"}
        dataset=SlipFrames([slip[0],second])
        selected=stratified_smoke_indices(dataset,8)
        self.assertEqual({dataset.index[index][2] for index in selected},{0,1})
        self.assertEqual({dataset.index[index][0] for index in selected},{0,1})


if __name__=="__main__":unittest.main()
