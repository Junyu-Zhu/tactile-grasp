#!/usr/bin/env python3
import argparse,csv,math
from collections import defaultdict
from pathlib import Path
import numpy as np
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows):
 with open(p,"w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--draws",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();groups=defaultdict(list)
 for r in read(a.draws):
  try:v=float(r["difference"])
  except (ValueError,TypeError):v=float("nan")
  groups[(int(r["fold"]),r["metric"])].append(v)
 out=[]
 for (fold,metric),v in sorted(groups.items()):
  z=np.asarray([x for x in v if math.isfinite(x)]);row={"fold":fold,"metric":metric,"requested_draws":len(v),"valid_draws":len(z),"support_status":"supported" if len(z) else "unsupported_no_required_class_in_validation_resamples"}
  row.update({"mean":float(z.mean()),"q025":float(np.quantile(z,.025,method="linear")),"median":float(np.quantile(z,.5,method="linear")),"q975":float(np.quantile(z,.975,method="linear"))} if len(z) else {"mean":"","q025":"","median":"","q975":""});out.append(row)
 write(a.output,out);print({"rows":len(out),"unsupported":sum(x["valid_draws"]==0 for x in out)})
if __name__=="__main__":main()
