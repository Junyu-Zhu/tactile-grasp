#!/usr/bin/env python3
import argparse,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 loader=unittest.TestLoader();suite=loader.discover(str(HERE),pattern='test_analyze_current.py');result=unittest.TextTestRunner(verbosity=2).run(suite)
 payload={'status':'pass' if result.wasSuccessful() else 'fail','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_hashes':{str(HERE/'analyze_current.py'):sha(HERE/'analyze_current.py'),str(HERE/'test_analyze_current.py'):sha(HERE/'test_analyze_current.py'),str(HERE/'PROTOCOL.md'):sha(HERE/'PROTOCOL.md'),str(HERE/'AMENDMENT_01.md'):sha(HERE/'AMENDMENT_01.md'),str(HERE/'AMENDMENT_02.md'):sha(HERE/'AMENDMENT_02.md')}}
 args.output.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=args.output.name+'.',suffix='.tmp',dir=str(args.output.parent))
 with os.fdopen(fd,'w') as f:json.dump(payload,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,args.output);print(json.dumps(payload,indent=2))
 if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
