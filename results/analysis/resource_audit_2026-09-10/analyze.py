import csv, re, pathlib, collections, math, json
root=pathlib.Path('results/analysis/resource_audit_2026-09-10')
s=list(csv.DictReader((root/'sacct_current_results.psv').open(),delimiter='|'))
byid={r['JobID']:r for r in s}
logs={}
for p in pathlib.Path('logs').rglob('*.out'):
 m=re.search(r'_(\d{6,})(?:_(\d+))?\.out$',p.name)
 if m: logs[m[1]+('_'+m[2] if m[2] else '')]=p
rows=[]
for r in s:
 if '.' in r['JobID']:continue
 p=logs.get(r['JobID'])
 if not p:continue
 t=p.read_text(errors='replace')
 def header(k):
  m=re.search('^'+re.escape(k)+r':\s*(.*)$',t,re.M);return m[1].strip() if m else ''
 emb=header('Embedding model'); scope=header('Task scopes'); task=header('Task names'); pref=header('Model prefix'); mode=header('Embedding modes'); split=header('Split variant').replace('loco','similarity')
 if not mode:mode='attention' if ('at_' in r['JobName'] or 'noln' in r['JobName']) else 'no_attention'
 if not scope:scope='real_case' if r['JobName'].startswith('rc42') else 'properties'
 if not split:split='similarity' if r['JobName'].startswith('o42l') else 'random'
 batch=byid.get(r['JobID']+'.batch',{})
 rss=float(batch.get('MaxRSS','').rstrip('K') or 0)/1048576
 req=float(r['ReqMem'].rstrip('K') or 0)/1048576
 if scope=='real_case':group='I'
 elif task=='binding_site':group=('A' if 'mlp' in pref else "A'") if mode=='attention' else ('G1' if 'mlp' in pref else 'G2' if 'xgb' in pref else 'G3')
 elif task=='ec_number':group=('D' if 'mlp' in pref else 'B') if mode=='attention' else 'H'
 elif task=='active_sites':group='C' if mode=='attention' else 'G3'
 else:group='F' if mode=='attention' else 'E'
 store=';'.join(dict.fromkeys(re.findall(r'Residue store auto-selection:.*',t)))
 resumes=';'.join(x.strip() for x in t.splitlines() if re.search('resum',x,re.I))
 rows.append(dict(job_id=r['JobID'],job_name=r['JobName'],group=group,embedding=emb,scope=scope,task=task,prefix=pref,mode=mode,split=split,state=r['State'],elapsed_h=int(r['ElapsedRaw'])/3600,maxrss_gib=rss,requested_gib=req,cpus=r['AllocCPUS'],log=str(p),store=store,resume=resumes))
with (root/'historical_jobs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
buckets=collections.defaultdict(list)
for r in rows:buckets[(r['group'],r['split'],r['embedding'])].append(r)
agg=[]
for (g,split,emb),rs in sorted(buckets.items()):
 h=sum(r['elapsed_h'] for r in rs); ram=max(r['maxrss_gib'] for r in rs)
 # An operational buffer, not a statistical prediction. Keep failures in totals.
 buffered=1.3*h+0.25
 band=next((x for x in [1,2,3,6,12,24,48,72] if x>=buffered),168)
 agg.append(dict(group=g,split=split,embedding=emb,jobs=len(rs),historical_total_h=round(h,4),historical_peak_gib=round(ram,3),requested_peak_gib=max(r['requested_gib'] for r in rs),has_failed_attempt=any(r['state']!='COMPLETED' for r in rs),heuristic_time_h=band,job_ids=';'.join(r['job_id'] for r in rs)))
with (root/'by_embedding.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
print('Jobs matched',len(rows),'outcomes',dict(collections.Counter(r['state'] for r in rows)))
for g in ['A',"A'",'B','D','C','G1','G2','G3','H','E','F','I']:
 a=[r for r in agg if r['group']==g]; v=sorted(r['historical_total_h'] for r in a); top=max(a,key=lambda x:x['historical_total_h'])
 print(g,'profiles',len(a),'hours range',round(min(v),2),round(max(v),2),'RAM',round(min(x['historical_peak_gib'] for x in a),1),round(max(x['historical_peak_gib'] for x in a),1),'top',top['embedding'],top['split'],'recommended bands',dict(collections.Counter(x['heuristic_time_h'] for x in a)))
print('Total historical allocated hours',sum(x['elapsed_h'] for x in rows))
