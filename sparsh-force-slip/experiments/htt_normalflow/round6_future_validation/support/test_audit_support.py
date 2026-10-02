from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

P=Path(__file__).with_name("audit_support.py")
S=importlib.util.spec_from_file_location("r6_support",P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)


class SupportAuditTests(unittest.TestCase):
    def test_htt_stops_at_first_gross_and_requires_complete_future(self):
        ts=np.arange(50);stage=np.zeros(50,dtype=np.int8);stage[22]=2
        onset,pos,neg,excluded=m.endpoint_trial(ts,stage,8,13,True)
        self.assertEqual(onset,22);self.assertEqual(pos,list(range(14,22)))
        self.assertEqual(neg,[13]);self.assertEqual(excluded["at_or_after_first_gross"],28)

    def test_source_uses_current_slip_one_as_onset(self):
        ts=np.arange(20);current=np.zeros(20,dtype=np.int8);current[10:]=1
        onset,pos,neg,_=m.endpoint_trial(ts,current,3,3,False,onset_label=1)
        self.assertEqual(onset,10);self.assertEqual(pos,[7,8,9]);self.assertEqual(neg,[3,4,5,6])

    def test_trial_threshold_counts_unique_leakage_groups(self):
        trials=[{"episode_id":"a","leakage_group":"g","positive_endpoints":[1,2],"negative_endpoints":[0],"excluded":{}},{"episode_id":"b","leakage_group":"g","positive_endpoints":[3],"negative_endpoints":[],"excluded":{}}]
        result=m.summarize_trials(trials)
        self.assertEqual(result["positive_endpoints"],3);self.assertEqual(result["positive_event_episodes"],2);self.assertEqual(result["positive_event_trials"],1)

    def test_deterministic_roles_are_disjoint_and_repeatable(self):
        groups=[f"g{i}" for i in range(20)]
        left=m.deterministic_roles(groups,"source");right=m.deterministic_roles(list(reversed(groups)),"source")
        self.assertEqual(left,right);self.assertEqual(sum(x=="selection" for x in left.values()),4);self.assertEqual(sum(x=="calibration" for x in left.values()),4)

    def test_actual_feature_history_uses_candidate_intersection(self):
        ts=np.arange(25);current=np.zeros(25,dtype=np.int8);current[20:]=1
        available=set(range(10,25));onset,pos,neg,excluded=m.endpoint_trial(ts,current,3,13,False,onset_label=1,candidate_ts=np.asarray(sorted(available)),history_ts=available)
        self.assertEqual(onset,20);self.assertEqual(pos,[17,18,19]);self.assertEqual(neg,[13,14,15,16]);self.assertEqual(excluded["incomplete_exact_history"],3)

    def test_htt_never_reads_all_test_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);label=root/"allowed.npy";np.save(label,np.zeros(30,dtype=np.int8))
            folds=[f"htt_leave_p{i}" for i in range(1,5)]
            allowed_roles={fold:"train" for fold in folds};test_roles={fold:"test" for fold in folds}
            split={"episodes":[{"id":"allowed","task":"slip","leakage_group":"g1"},{"id":"always-test","task":"slip","leakage_group":"g2"}],"splits":{fold:{"train":["allowed"],"validation":[],"calibration":[],"test":["always-test"]} for fold in folds}}
            split_path=root/"splits.json";split_path.write_text(json.dumps(split))
            contract={"format":"round5_htt_mae_tokens_targets_v1","status":"complete","split_manifest_sha256":m.sha256(split_path),"entries":[{"episode_id":"allowed","task":"slip","frames":30,"roles_by_fold":allowed_roles,"label_path":str(label),"label_sha256":m.sha256(label)},{"episode_id":"always-test","task":"slip","frames":30,"roles_by_fold":test_roles,"label_path":str(root/"must-not-open.npy"),"label_sha256":"0"*64}]}
            contract_path=root/"contract.json";contract_path.write_text(json.dumps(contract));protocol=json.loads(m.PROTOCOL.read_text())
            result=m.audit_htt(contract_path,split_path,protocol)
            self.assertEqual(result["label_access"]["always_test_episode_count"],1);self.assertFalse(result["label_access"]["always_test_labels_read"])


if __name__=="__main__":unittest.main()
