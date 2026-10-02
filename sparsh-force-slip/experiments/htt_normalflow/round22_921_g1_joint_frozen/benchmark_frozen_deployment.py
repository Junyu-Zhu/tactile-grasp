#!/usr/bin/env python3
import argparse,json,time,torch,datetime,subprocess,statistics
from pathlib import Path
import train_frozen as D
import train_f1 as F

def bench(model,x,batch,blocks=5,iters=50):
 z=x[:batch].cuda();model=model.cuda().eval()
 with torch.inference_mode():
  for _ in range(30):model(z)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
  for _ in range(blocks):
   t=time.perf_counter()
   for _ in range(iters):model(z)
   torch.cuda.synchronize();times.append((time.perf_counter()-t)/iters*1000)
 return {'batch':batch,'blocks':blocks,'iterations_per_block':iters,'mean_batch_ms':statistics.mean(times),'std_batch_ms':statistics.pstdev(times),'min_batch_ms':min(times),'max_batch_ms':max(times),'mean_endpoint_us':statistics.mean(times)/batch*1000,'peak_cuda_allocated_bytes_process':torch.cuda.max_memory_allocated()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--physical-gpu-index',type=int,required=True);ap.add_argument('--gpu-uuid',required=True);a=ap.parse_args();inv=json.loads(a.inventory.read_text())['runs'];rows=[]
 for group in ('V','C','M','MB','K-V','K-F','K-VF'):
  rr=next(r for r in inv if r['group']==group);out=Path(rr['output']);cm=json.loads((out/'COMMIT.json').read_text());ck=torch.load(out/cm['best']['path'],map_location='cpu',weights_only=False);d=torch.load(rr['data'],map_location='cpu',weights_only=False)
  if rr['package'] in ('E1','E2'):
   n=ck['normalizer'];x=(d['roles']['validation']['x']-n['mean'])/n['std'];m=D.init_model(group,rr['seed'])
  else:
   n=ck['normalizer'];x=F.normalize_x(d['roles']['validation']['x'],n,group);m=F.init_model(group,rr['seed'])
  m.load_state_dict(ck['model']);params=sum(q.numel() for q in m.parameters());trainable=sum(q.numel() for q in m.parameters() if q.requires_grad)
  for batch in (1,256):rows.append({'group':group,'package':rr['package'],'new_module_parameters':params,'trainable_parameters':trainable,**bench(m,x,batch)})
  del m,x,d,ck;torch.cuda.empty_cache()
 out={'schema':'round22_frozen_deployment_cost_v1','status':'pass','measured_at':datetime.datetime.now().astimezone().isoformat(),'device':torch.cuda.get_device_name(0),'physical_gpu_index':a.physical_gpu_index,'gpu_uuid':a.gpu_uuid,'torch':torch.__version__,'cuda':torch.version.cuda,'shared_gpu_process_snapshot':subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True,capture_output=True).stdout.splitlines(),'rows':rows,'reuse_boundary':{'reused':'historical frozen visual prefix features and frozen current-force anchor inputs','new':'E1/E2 recurrent slip heads and F1 recurrent delta heads; F2 is scalar multiply with no parameters','excluded_from_latency':'raw tactile decoding, frozen visual prefix extraction, historical current-force inference, disk I/O, continuous alarm state machine','deployment_claim':'head-only incremental cost; not end-to-end sensor latency'},'measurement':'CUDA synchronized, inference mode, 30 warmups, five blocks of 50 iterations; min/max/std are block mean latency','memory_note':'peak_cuda_allocated_bytes_process is the PyTorch process allocation peak during measurement, not exclusive board memory','test_consumed':False};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','rows':len(rows),'device':out['device']}))
if __name__=='__main__':main()
