#!/usr/bin/env python3
"""Reconstruct the exact seed-42 raw-inference encoder and bind it to cached MAE."""
import argparse,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE.parents[2]/'scripts'))
import phase2_b_multitask as p2
sys.path.insert(0,str(HERE/'current'))
from common import tensor_state_sha256,sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.manual_seed(42);model=p2.FrozenEncoderSharedForceSlip('mae',decoder_variant='decoupled')
 weight_sha=sha256_file(p2.ENCODER_CHECKPOINTS['mae']);state_sha=tensor_state_sha256(model.encoder.state_dict())
 checks={'canonical_weight':weight_sha=='83ae4a8fa6a7bbd9702b14586feafdba63252e83afad26c3e5832e0ad39446ad','constructed_state_matches_frozen_cache':state_sha=='60ab1af933e4c9a6791e8090554ded0f2229f09c5303a2a2a97ba66922c1eea8','eval_mode':not model.encoder.training,'gradient_disabled':all(not x.requires_grad for x in model.encoder.parameters())}
 result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'weight_path':str(p2.ENCODER_CHECKPOINTS['mae']),'weight_sha256':weight_sha,'encoder_state_sha256':state_sha,'constructor_source_sha256':sha256_file(Path(p2.__file__)),'seed':42,'load_info':model.load_info,'scope':'same constructor/seed used by fresh source regression and e2e benchmark; independent CPU identity reconstruction'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
