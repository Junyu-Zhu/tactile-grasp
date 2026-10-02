#!/usr/bin/env python3
"""CPU-only verification that two trainers cannot own the same output."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
import train_e3
def owner(output,ready):
 h=train_e3.acquire_task_lock(output);ready.set();time.sleep(1);h.close()
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--receipt",type=Path,required=True);a=p.parse_args();ready=mp.Event();proc=mp.Process(target=owner,args=(a.output,ready));proc.start();assert ready.wait(10);rejected=False
 try:train_e3.acquire_task_lock(a.output)
 except RuntimeError:rejected=True
 proc.join(10);result={"schema":"round22_task_lock_smoke_v1","status":"pass" if rejected and proc.exitcode==0 else "fail","second_owner_rejected":rejected,"first_owner_exitcode":proc.exitcode,"neural_training_started":False};a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
