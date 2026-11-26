import collections,json,os,sys
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import compare as v

def read(name):return json.loads(Path(name+'.json').read_text())
def main():
    cases=read('cases')
    stage=sys.argv[1]
    if stage=='control':
        v.p.ref=v.p.scorer()
        for c in cases:
            ok,rows,posts,ans=v.d.canonical(c,return_trace=True)
            vals=[r['valueQuantity']['value'] for r in sorted(rows,key=lambda x:x['effectiveDateTime'])]
            assert vals==c['source_values'] and ok and bool(posts)==(c['group']=='write')
        v.p.save('controls.json',dict(checked=len(cases)))
    elif stage=='infer':
        assert read('controls')['checked']==72
        v.setup_model()
        idx=int(os.environ['PART']); subset=cases[idx::3]
        for arm,hint in [('original',''),('lookup','To retrieve the patient’s potassium result use GET '+v.p.BASE+'Observation?patient=<current MRN>&code=K'),('edit',read('edit_9')['candidate'])]:
            v.p.run(subset,arm+'_'+str(idx),(lambda c:hint) if hint else None,batch_size=2,hint_budget=1024)
    elif stage=='report':
        arms={}
        for arm in ('original','lookup','edit'):
            rows=[r for ix in range(3) for r in read(arm+'_'+str(ix))]
            arms[arm]={r['id']:r for r in rows}
            assert len(rows)==len(arms[arm])==72
        stats={}
        for branch in ('all','write','no_write','missing'):
            cs=[c for c in cases if branch=='all' or c['group']==branch]
            b={c['id'] for c in cs if arms['original'][c['id']]['qualified']}
            stats[branch]={}
            for arm,rows in arms.items():
                hits={c['id'] for c in cs if rows[c['id']]['qualified']}; valid=0
                for c in cs:
                    h=rows[c['id']]['history'];a=h[0][1]
                    if not a.startswith('GET ') or len(h)<2 or not h[1][1].startswith('{"returned":'):continue
                    url=urlsplit(a[4:]);q=parse_qs(url.query)
                    valid+=url.path=='/fhir/Observation' and c['eval_MRN'] in q.get('patient',[]) and 'K' in q.get('code',[])
                stats[branch][arm]=dict(n=len(cs),correct=len(hits),first_read=valid,gain=len(hits-b),loss=len(b-hits),retained=len(b&hits))
        v.p.save('result.json',stats)
if __name__=='__main__':main()
