"""Check old probabilities on all cached tokens and sampled direct encoder inputs."""
import json
import os
from pathlib import Path
import numpy as np
import torch
import cache as c

OUT = Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round3_mae_slip_adaptation')


def main():
    c.configure_determinism()
    cm = json.loads((OUT/'cache/cache_manifest.json').read_text())
    old = json.loads(Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2/cache/index.json').read_text())
    old_rows = {r['id']:r for r in old['episodes']}
    manifest = json.loads(Path(cm['manifest']).read_text())
    source = {r['id']:r for r in manifest['episodes']}
    model, _ = c.p2.load_b_checkpoint(Path(cm['checkpoint']), torch.device('cuda:0'))
    model.eval().requires_grad_(False)
    report = {'atol':1e-4,'rtol':1e-3,'episodes':[], 'direct_tokens':[]}
    with torch.inference_mode():
        for i, entry in enumerate(cm['entries']):
            tokens=np.load(entry['token_path'],mmap_mode='r')
            predictions=[]
            for start in range(0,len(tokens),128):
                z=torch.from_numpy(np.array(tokens[start:start+128],copy=True)).cuda()
                predictions.append(model.decoder(z)['slip'].softmax(1)[:,1].cpu().numpy())
            candidate=np.concatenate(predictions)
            old_path=Path(old_rows[entry['episode_id']]['cache_path'])
            if c.sha256_file(old_path)!=old_rows[entry['episode_id']]['cache_sha256']:
                raise ValueError('Changed old cache')
            with np.load(old_path) as data:
                baseline=data['p_slip']
                assert np.array_equal(data['labels'],np.load(entry['label_path']))
            ok=np.allclose(candidate,baseline,atol=1e-4,rtol=1e-3)
            report['episodes'].append({'id':entry['episode_id'],'frames':len(tokens),'pass':bool(ok),
                                       'max_abs':float(np.max(np.abs(candidate-baseline))),
                                       'decision_changes_at_0_5':int(np.sum((candidate>=.5)!=(baseline>=.5)))})
            if i in (0,len(cm['entries'])//2,len(cm['entries'])-1):
                episode=c.adapters.load_htt(source[entry['episode_id']]['path'])
                idx=[0,min(6,len(tokens)-1)]
                images=torch.stack([c.adapters.window(episode,t)['inputs']['image'] for t in idx]).cuda()
                direct=model.encoder(images).cpu().numpy()
                expected=np.array(tokens[idx])
                report['direct_tokens'].append({'id':entry['episode_id'],'frames':idx,
                                               'pass':bool(np.allclose(direct,expected,atol=1e-4,rtol=1e-3)),
                                               'max_abs':float(np.max(np.abs(direct-expected)))})
    report['status']='pass' if all(r['pass'] for r in report['episodes']+report['direct_tokens']) else 'fail'
    report['max_probability_abs_difference']=max(r['max_abs'] for r in report['episodes'])
    report['decision_changes_at_0_5']=sum(r['decision_changes_at_0_5'] for r in report['episodes'])
    (OUT/'cache_baseline_equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','direct_tokens')}))
    if report['status']!='pass': raise SystemExit(1)


if __name__=='__main__': main()
