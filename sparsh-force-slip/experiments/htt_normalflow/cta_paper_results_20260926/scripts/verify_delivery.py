from pathlib import Path
import csv,json,hashlib,re,collections
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((P/'tables'/n).open()))
checks={}
for name in ['FIGURE_MANIFEST.json','REPORT_SOURCES.json']:
 m=json.loads((P/name).read_text())
 for item in m['inputs']:
  q=Path(item['path']);assert sha(q)==item['sha256'],str(q)
 assert m['exclusions']==[]
checks['source_hashes']=True
m=json.loads((P/'FIGURE_MANIFEST.json').read_text())
assert sha(P/'scripts/build_figures.py')==m['script_sha256']
assert len(m['figures'])==21
for x in m['figures']:assert sha(P/x['path'])==x['sha256']
checks['seven_figures_three_formats']=True
r=rows('RQ1_all_runs.csv');assert len(r)==96
c=collections.Counter((x['stage'],x['group']) for x in r);assert len(c)==8 and set(c.values())=={12}
assert len({(x['stage'],x['group'],x['fold'],x['seed']) for x in r})==96
r=rows('RQ2_all_runs.csv');assert len(r)==216
c=collections.Counter((x['variant'],x['horizon']) for x in r);assert len(c)==18 and set(c.values())=={12}
assert len(rows('CONDITIONAL_H10_ALL_STRATA.csv'))==20
assert len(rows('historical_R8_H3_operating_points.csv'))==8
checks['all_groups_folds_seeds_retained']=True
a=json.loads((P/'audit/MACHINE_AUDIT.json').read_text());assert a['passed'] and not a['problems']
assert a['scope']['all_run_metadata']==132 and a['scope']['actual_checkpoint_tensor_samples']==44 and a['scope']['raw_endpoint_alignment_samples']==16
checks['server_audit_passed']=True
s=(P/'CTA论文实验结果.md').read_text();assert '<!-- AUDIT_BLOCK -->' not in s and '服务器实体审计进行中' not in s
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s):
 q=Path(target);q=q if q.is_absolute() else P/q
 assert q.exists(),str(q)
assert len(re.findall(r'!\[',s))==7
checks['report_links_and_figures']=True
result={'passed':True,'checks':checks,'scope':'Source hashes, complete table grids, 7 figure files in 3 formats, machine audit and report links. Scientific claims use existing paired CI, no new training/inference or unfavorable exclusions.','report_sha256':sha(P/'CTA论文实验结果.md')}
(P/'AUTHOR_DELIVERY_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
