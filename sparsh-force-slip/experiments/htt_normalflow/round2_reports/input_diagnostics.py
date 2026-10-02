"""Fixed validation examples showing raw/reference/legacy encoder inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters import load_htt,load_normalflow,preprocess
from prepare_splits import file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text());rows={r['id']:r for r in manifest['episodes']}
    htt=[rows[i] for i in manifest['splits']['htt_leave_p1']['validation'] if rows[i]['task']=='slip'][:2]
    nf=[];seen=set()
    for i in manifest['splits']['normalflow_objects']['validation']:
        row=rows[i]
        if row['group'] not in seen: nf.append(row);seen.add(row['group'])
    chosen=htt+nf;fig,axes=plt.subplots(len(chosen),3,figsize=(10,3*len(chosen)));records=[]
    for k,row in enumerate(chosen):
        for path,expected in row['source_files'].items():
            if file_hash(path)!=expected:raise ValueError('Source changed')
        e=load_htt(row['path']) if row['domain']=='htt' else load_normalflow(row['path'])
        t=min(50,len(e.images)-1)
        transformed=preprocess(e.images[t],e.reference).permute(1,2,0).numpy()
        if e.reference is not None:axes[k,0].imshow(np.clip(e.reference/255,0,1))
        else:axes[k,0].text(.5,.5,'No unloaded reference supplied',ha='center',va='center',wrap=True)
        axes[k,0].set_title('Supplied reference')
        axes[k,1].imshow(e.images[t]);axes[k,1].set_title(f'Raw RGB frame {t}')
        axes[k,2].imshow(transformed);axes[k,2].set_title('Legacy encoder preprocessing')
        axes[k,0].set_ylabel(e.episode_id)
        for ax in axes[k]:ax.set_xticks([]);ax.set_yticks([])
        records.append({'episode_id':e.episode_id,'role':'htt_leave_p1/validation' if row['domain']=='htt' else 'normalflow_objects/validation',
                        'frame_index':t,'raw_range':[int(e.images[t].min()),int(e.images[t].max())],
                        'preprocessed_range':[float(transformed.min()),float(transformed.max())],
                        'background_subtraction':e.reference is not None,'source_files':row['source_files']})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(out/'input_domain_examples.png',dpi=140);plt.close(fig)
    (out/'input_domain_examples.json').write_text(json.dumps({'selection':'first two HTT validation slip episodes; first trial of each NF validation object; fixed frame min(50,T-1)',
        'examples':records,'interpretation':'HTT uses raw-reference subtraction; NormalFlow raw images have no supplied unloaded reference. Visual differences are preprocessing/domain evidence, not performance proof.'},indent=2))

if __name__=='__main__':main()
