#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'encoders'))
import cache

root=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
out=root/'round4_comprehensive/diagnostics'
manifest=json.loads((root/'round1/splits.json').read_text())
row=next(r for r in manifest['episodes'] if r['id']=='htt/p1_sliding/0_press_7')
ep=cache.adapters.load_htt(row['path'])
fig,axes=plt.subplots(1,3,figsize=(10,5))
images=[ep.images[37],cache.encoder_input(ep,37,'dino')[:3].permute(1,2,0).numpy(),
        cache.encoder_input(ep,37,'mae_letterbox')[:3].permute(1,2,0).numpy()]
for ax,img,title in zip(axes,images,['Raw 224 x 224 (existing black bands)','Legacy crop + resize 320 x 240','Reference difference + aspect-preserving fit']):
    ax.imshow(img);ax.set_title(title,fontsize=9);ax.axis('off')
fig.tight_layout();fig.savefig(out/'geometry_comparison.png',dpi=160)
with (out/'REPORT_ZH.md').open('a') as f:
    f.write('\n原始图像本身已有上下黑边；旧裁剪可能部分去除这些黑边，因而不能把“裁剪存在”直接当作错误。可确定的是几何缩放不等比。新方案同时保留原始视野和黑边，其收益或损失须由预定对照决定。geometry_comparison.png 展示同一帧的三种形态。\n')
