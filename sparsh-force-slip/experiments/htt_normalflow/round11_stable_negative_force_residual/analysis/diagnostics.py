"""Pure array diagnostics, no model fitting or threshold selection."""
import numpy as np

def correctness(stage,base_probability,full_probability,threshold):
 s=np.asarray(stage);b=np.asarray(base_probability)>=threshold;f=np.asarray(full_probability)>=threshold;primary=s!=1;y=s==2
 return {'corrected':primary&(b!=y)&(f==y),'harmed':primary&(b==y)&(f!=y),'static_corrected':(s==0)&b&~f,'static_harmed':(s==0)&~b&f,'gross_corrected':(s==2)&~b&f,'gross_harmed':(s==2)&b&~f}

def residual_stats(values,bound):
 v=np.asarray(values,dtype=float)
 if not np.isfinite(v).all() or bound<=0 or np.max(abs(v),initial=0)>bound+1e-5:raise ValueError('invalid bounded residual')
 return dict(n=len(v),mean=float(v.mean()) if len(v) else None,mean_abs=float(abs(v).mean()) if len(v) else None,negative=int((v<0).sum()),zero=int((v==0).sum()),positive=int((v>0).sum()),saturated=int((abs(v)>=.95*bound).sum()))

def sequence_bin(t,length):
 if length<=1 or not 0<=t<length:raise ValueError('bad sequence frame')
 return min(3,int(4*t/(length-1)))
