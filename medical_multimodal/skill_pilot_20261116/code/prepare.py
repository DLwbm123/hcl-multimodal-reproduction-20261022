"""Group-isolated pilot roles; original tests remain untouched."""
import json,os,random
from pathlib import Path
R=Path(os.environ['RUN_ROOT']); H=R.parent; D=H/'medical_transfer_v1/prepared_data'; A=Path(os.environ['DATA_ROOT'])
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
roles={r['id']:r for r in rows(D/'audit/roles.jsonl')}; used=set(); seen=set(); train=set(); val=set(); confirmation=set()
def collect(o,out):
 if isinstance(o,dict):
  if o.get('group_id'):out.add(o['group_id'])
  for k in ['id','task_id','internal_sample_id','case_id']:
   if isinstance(o.get(k),str) and o[k] in roles:out.add(roles[o[k]]['group_id'])
  for v in o.values():collect(v,out)
 elif isinstance(o,list):
  for v in o:collect(v,out)
def read(p):
 p=Path(p)
 if str(p) in seen:return
 assert p.exists(),str(p)
 seen.add(str(p));o=rows(p) if p.suffix=='.jsonl' else json.loads(p.read_text());collect(o,used)
 if isinstance(o,dict):
  for s in o.get('exclusion_sources',[]):read(s)
for pat in ['agent_*/private_membership.json','vqa_*/private_membership.json','agent_*/private_learning.json','vqa_*/private_learning.json']:
 for p in sorted(H.glob(pat)):read(p)
for pat in ['agent_*/private_learning.json','vqa_*/private_learning.json']:
 for p in sorted(H.glob(pat)):
  o=json.loads(p.read_text());collect(o.get('train',[]),train);collect(o.get('selection',[]),val);collect(o.get('development',[]),val)
for pat in ['agent_*/private_confirmation.json','vqa_*/private_confirmation.json']:
 for p in sorted(H.glob(pat)):collect(json.loads(p.read_text()),confirmation)
visible={r['id']:r for r in rows(D/'visible/polyp_localization/h_search.jsonl')}
targets={r['case_id']:r for r in rows(A/'private/targets/polyp_localization/adaptation.jsonl')}
bygroup={}
for r in rows(D/'private/polyp_localization/h_search.jsonl'):
 if r['id'] in targets:bygroup.setdefault(r['metadata']['group_id'],[]).append(r)
rng=random.Random(20261116)
def pick(gs,n):
 gs=sorted(gs);rng.shuffle(gs);assert len(gs)>=n,(len(gs),n);return gs[:n]
tg=pick((set(bygroup)&train)-val-confirmation,14)
vg=pick((set(bygroup)&val)-train-confirmation,12)
cg=pick(set(bygroup)-used,8)
parts={'preflight':tg[:2],'train':tg[2:],'validation':vg,'confirmation':cg};data={}
for role,gs in parts.items():
 data[role]=[]
 for g in gs:
  r=sorted(bygroup[g],key=lambda x:x['id'])[0];v=visible[r['id']];t=targets[r['id']]['target']
  x={'id':r['id'],'group_id':g,'image':v['image'],'mask':str(A/t['mask']),'bbox':t['bbox'],'width':r['metadata']['width'],'height':r['metadata']['height'],'role':role}
  assert Path(x['image']).is_file() and Path(x['mask']).is_file()
  data[role].append(x)
assert len(set(sum(parts.values(),[])))==34
(R/'private_data.json').write_text(json.dumps({k:v for k,v in data.items() if k!='confirmation'},indent=2))
(R/'private_confirmation.json').write_text(json.dumps(data['confirmation'],indent=2))
(R/'private_membership.json').write_text(json.dumps({'seed':20261116,'roles':parts,'exclusion_sources':sorted(seen),'training_validation_reused_development_only':True,'confirmation_unseen_in_audited_history':True,'patient_independence':'unverified'},indent=2))
(R/'DATA_CHECK.json').write_text(json.dumps({'counts':{k:len(v) for k,v in data.items()},'source_groups_disjoint':True,'original_test_untouched':True,'confirmation_exclusion_sources':len(seen),'tasks_share_roles':True},indent=2))
print((R/'DATA_CHECK.json').read_text())
