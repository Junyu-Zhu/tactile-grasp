from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

import analysis_common as c
import analyze_force
import export_failures
import sensitivity


class AnalysisTests(unittest.TestCase):
    def test_regression_and_exact_lag(self):
        y=np.array([[0,0,1],[1,0,2],[2,0,3],[3,0,4],[4,0,5],[5,0,6]],np.float32);p=y+1
        m=c.regression_metrics(y,p);self.assertEqual(m["axis"]["rmse_n"],[1,1,1])
        lag=c.lagged_change_metrics(np.array(["e"]*6),np.arange(5,11),y,p,5)
        self.assertEqual(lag["count"],1);self.assertEqual(lag["axis"]["rmse_n"],[0,0,0])

    def test_mismatch_preserves_rows(self):
        x=np.arange(20).reshape(4,5);z=c.circular_mismatch(x)
        self.assertFalse(np.array_equal(x,z));self.assertEqual(sorted(map(tuple,x)),sorted(map(tuple,z)))

    def test_sensitivity_sort_is_explicit(self):
        tokens=np.asarray([[[3]],[[1]],[[2]]]);conditions=[np.asarray([3]),np.asarray([1]),np.asarray([2])]
        got=sensitivity.sort_examples(tokens,conditions,[0,1,0],["b","a","a"],[1,2,1])
        self.assertEqual(got[3],["a","a","b"]);self.assertEqual(got[4],[1,2,1]);self.assertEqual(got[0][:,0,0].tolist(),[2,1,3])

    def test_inventory_receipt_is_exactly_bound(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);summary=d/"training_summary.json";summary.write_text("{}")
            source=d/"input.json";source.write_text("stable")
            argv=["python","train.py","--output",str(d)]
            receipt=summary.with_name(summary.name+".scheduler_receipt.json")
            receipt.write_text(json.dumps({"argv":argv,"input_sha256":{str(source):c.sha256(source)}}))
            inventory=d/"inventory.json";inventory.write_text(json.dumps({"jobs":[{"acceptance_path":str(summary),"argv":argv}]}))
            proof=c.verify_receipt_inventory(summary,inventory);self.assertEqual(proof["inventory_sha256"],c.sha256(inventory))
            source.write_text("drift")
            with self.assertRaisesRegex(ValueError,"input drift"):c.verify_receipt_inventory(summary,inventory)

    def test_cluster_bootstrap_is_deterministic(self):
        groups=np.asarray(["a","a","b","b"]);target=np.asarray([[0,0,1],[0,1,1],[1,0,2],[1,1,2]],np.float32)
        pred=target+.25;mean=np.zeros(3,np.float32)
        one=c.cluster_force_bootstrap(groups,target,pred,mean);two=c.cluster_force_bootstrap(groups,target,pred,mean)
        self.assertEqual(one,two);self.assertEqual(one["unit"],"complete schema-2 leakage_group")

    def test_source_cache_regression_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);checkpoint=d/"model.pth";checkpoint.write_bytes(b"x");features=d/"features.pt";out=d/"metrics.json"
            torch.save({"force_gt_n":torch.zeros(3,3),"force_pred_n":torch.ones(3,3),"metadata":[{}]*3,"checkpoint":str(checkpoint),"encoder_checkpoint_payload_epoch":30,"split":"val"},features)
            with mock.patch.object(analyze_force,"protocol",return_value={"canonical_source_checkpoint_sha256":c.sha256(checkpoint)}):
                result=analyze_force.evaluate_source(argparse.Namespace(features=features,source_checkpoint=checkpoint,output=out))
            self.assertEqual(result["regression"]["axis"]["rmse_n"],[1,1,1]);self.assertEqual(result["source_epoch"],30)

    def test_failure_selection_one_per_episode(self):
        rows=[{"episode_id":"a","stage":0,"probability":.9},{"episode_id":"a","stage":0,"probability":.8},{"episode_id":"b","stage":0,"probability":.7}]
        selected=export_failures.select(rows,0,True,12)
        self.assertEqual([x["episode_id"] for x in selected],["a","b"])


if __name__=="__main__":unittest.main()
