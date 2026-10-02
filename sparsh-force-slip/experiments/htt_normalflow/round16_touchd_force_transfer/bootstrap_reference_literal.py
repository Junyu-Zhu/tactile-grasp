"""Frozen small-draw literal reference for the pre-optimization slip bootstrap loop."""
import numpy as np

def literal(by,candidate,base,seeds,metrics,samples,aggregate,weights,convert):
 out={metric:[] for metric in metrics}
 for metric in metrics:
  for sample in samples:
   w=weights(sample);diff=[]
   for seed in seeds:
    ca=aggregate([convert(r) for r in by[(candidate,seed)]],w)
    ba=aggregate([convert(r) for r in by[(base,seed)]],w)
    diff.append(ca[metric]-ba[metric])
   out[metric].append(float(np.mean(diff)))
 return out
