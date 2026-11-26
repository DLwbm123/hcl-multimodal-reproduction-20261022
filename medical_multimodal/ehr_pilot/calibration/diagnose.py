"""Development-only simulated EHR execution and paired calibration."""
import ast
import collections
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

# Keep this module next to the original runner when installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pilot as p

CODES = {4:'MG',5:'MG',6:'GLU',7:'GLU',9:'K',10:'A1C'}
WRITES = {3,8}
MONITOR = datetime.fromisoformat('2023-11-13T10:15:00+00:00')
PRIOR = Path(os.environ['PRIOR'])
DOCS = json.loads((p.DATA/'funcs_v1.json').read_text())
ROUTES = {f['name'].split('/')[-1] for f in DOCS if f['name'].startswith('GET ')}
GUIDES = {
    1:'GET <base>/Patient?given=<first name>&family=<last name>&birthdate=<DOB>. Match the requested name and date; read the identifier value and FINISH with a one-element JSON list. Use the requested missing-patient output if no match.',
    2:'GET <base>/Patient?identifier=<MRN>. Read birthDate. At the date in context subtract birth year from current year, subtract one more if the birthday is still ahead. FINISH([integer]).',
    3:'This is a new write. POST <base>/Observation on its own line followed by a complete JSON object on the next line. Fill category.coding from the vital-signs schema, code.text from the flowsheet ID, subject.reference from this patient, status, effectiveDateTime and valueString from this task. Wait for the simulated receipt before FINISH([]).',
    4:'GET <base>/Observation?patient=<MRN>&code=<lab code>. Inspect effectiveDateTime against the 24-hour window ending at the time in context. Take the latest qualifying value. FINISH with the value in a JSON list or the specified missing-value sentinel. Do not POST.',
    5:'GET <base>/Observation?patient=<MRN>&code=<lab code>. Find the latest in-window measurement. Use the threshold and dosing tiers in the current context. Only when replacement is required, POST the MedicationRequest with current subject, drug coding, timestamp, dose, route and rate. After the required actions FINISH with a JSON list.',
    6:'GET <base>/Observation?patient=<MRN>&code=<lab code>. Keep all measurements in the preceding 24 hours as defined by the current context. Compute sum divided by count; apply the specified missing-value rule if count is zero. FINISH([number]). Do not POST.',
    7:'GET <base>/Observation?patient=<MRN>&code=<lab code>. Compare timestamps across returned values. FINISH with the latest value in a JSON list; if absent apply the missing-value convention. Do not POST.',
    8:'Start a fresh referral. POST <base>/ServiceRequest then newline then complete JSON. Fill subject.reference, referral coding, authoredOn, status, intent, priority and note.text with the requested referral text. An example does not create an order. FINISH([]) only after the current request receipt.',
    9:'GET <base>/Observation?patient=<MRN>&code=<lab code>. Select the latest potassium. Inspect task conditions before writing. If low, construct the replacement MedicationRequest, then the follow-up ServiceRequest with the requested time. Use identifiers and codes from the current task only. If no order indicated, finish after checking the record.',
    10:'GET <base>/Observation?patient=<MRN>&code=<lab code>. Inspect the most recent value and timestamp. Check recency at the context date. If absent or older than the stated interval POST a ServiceRequest for the requested lab. Return the requested value and timestamp list, or the specified sentinel.'
}


def kind(case):
    return int(p.family(case)[4:])


def query(case):
    n=kind(case)
    if n==1:
        name,dob=re.search(r'name (.+?) and DOB of ([\d-]+)',case['instruction']).groups()
        first,last=name.split(' ',1)
        route,params='Patient',dict(given=first,family=last,birthdate=dob)
    else:
        # Only the identifier visible in the question is allowed into the input.
        mrn=re.search(r'\bS\d+\b',case['instruction']).group()
        route,params=('Patient',dict(identifier=mrn)) if n==2 else ('Observation',dict(patient=mrn,code=CODES[n]))
    return p.BASE+route+'?'+urlencode(params)


def fetch(url):
    address=urlsplit(url)
    route=address.path.removeprefix('/fhir/')
    params=parse_qs(address.query)
    if not url.startswith(p.BASE) or route not in ROUTES:
        return 'Invalid endpoint. Use a documented absolute resource URL.',None
    scope=('patient','subject','identifier','name','given','family','birthdate')
    if not any(params.get(k) for k in scope):
        return 'Search needs a patient-scoping argument.',None
    if any(k.startswith('_') and k not in ('_count','_format','_sort') for k in params):
        return 'Unsupported search control.',None
    params.update(_count=['256'],_format=['json'])
    page=p.BASE+route+'?'+urlencode(params,doseq=True)
    visited=set()
    rows=[]
    while page:
        target=urlsplit(page)
        origin=urlsplit(p.BASE)
        if ((target.scheme,target.netloc)!=(origin.scheme,origin.netloc)
                or not (target.path==origin.path.rstrip('/') or target.path.startswith(origin.path))
                or page in visited or len(visited)>=32):
            raise RuntimeError('pagination endpoint, cycle, or page limit')
        visited.add(page)
        res=p.session.get(page,timeout=25,allow_redirects=False)
        if res.status_code>=500:
            res.raise_for_status()
        if res.status_code!=200:
            return 'FHIR HTTP '+str(res.status_code),None
        bundle=res.json()
        if bundle.get('resourceType')!='Bundle':
            return 'Search did not return a Bundle.',None
        for entry in bundle.get('entry',[]):
            x=entry['resource']
            if x['resourceType']=='Patient':
                rows.append({k:x.get(k) for k in ('identifier','name','birthDate')})
            elif x['resourceType']=='Observation':
                rows.append({k:x[k] for k in ('subject','code','effectiveDateTime','valueQuantity','valueString') if k in x})
            else:
                rows.append(x)
        nxt=next((l['url'] for l in bundle.get('link',[]) if l.get('relation')=='next'),None)
        page=urljoin(p.BASE,nxt) if nxt else None
    rows.sort(key=lambda x:x.get('effectiveDateTime',''),reverse=True)
    return json.dumps(dict(returned=len(rows),rows=rows),separators=(',',':')),(route,parse_qs(address.query))


def model_fetch(url):
    payload,read=fetch(url)
    if not read or read[0]!='Observation':
        return payload,read
    rows=json.loads(payload)['rows']
    # Store the keys shared by observations once, without clipping rows.
    series={}
    for row in rows:
        value=row.get('valueQuantity',{})
        identity=(row.get('subject',{}).get('reference'), json.dumps(row.get('code',{}),sort_keys=True),value.get('unit'))
        series.setdefault(identity,[]).append([row.get('effectiveDateTime'),value.get('value',row.get('valueString'))])
    compact=[dict(subject=s,code=json.loads(c),unit=u,measurements=points) for (s,c,u),points in series.items()]
    return json.dumps(dict(returned=len(rows),columns=['effectiveDateTime','value'],series=compact),separators=(',',':')),read


def canonical(case, return_trace=False):
    n=kind(case)
    q=case['instruction']
    ctx=case['context']
    patient=re.search(r'\bS\d+\b',q)
    subject={'reference':'Patient/'+patient.group()} if patient else {}
    at=MONITOR.isoformat()
    posts=[]
    rows=[]
    reads=[]
    if n not in WRITES:
        data,r=fetch(query(case))
        assert r is not None
        rows=json.loads(data)['rows']
        reads=[r]
    measures=sorted([(datetime.fromisoformat(x['effectiveDateTime']),x['valueQuantity']['value'],x['effectiveDateTime']) for x in rows if 'valueQuantity' in x],reverse=True)
    if n in (4,5,6):
        measures=[m for m in measures if MONITOR-timedelta(days=1)<=m[0]<=MONITOR]
    last=measures[0][1] if measures else None
    def service(system,code):
        return dict(resourceType='ServiceRequest',code={'coding':[dict(system=system,code=code)]},authoredOn=at,status='active',intent='order',priority='stat',subject=subject)
    def medication(ndc,route,doserate):
        return dict(resourceType='MedicationRequest',medicationCodeableConcept={'coding':[dict(system='http://hl7.org/fhir/sid/ndc',code=ndc)]},authoredOn=at,status='active',intent='order',subject=subject,dosageInstruction=[dict(route=route,doseAndRate=[doserate])])
    answer=[]
    if n==1:
        answer=[rows[0]['identifier'][0]['value']] if rows else ['Patient not found']
    elif n==2:
        dob=date.fromisoformat(rows[0]['birthDate'])
        answer=[MONITOR.year-dob.year-((MONITOR.month,MONITOR.day)<(dob.month,dob.day))]
    elif n==3:
        posts.append(dict(resourceType='Observation',status='final',category=[{'coding':[dict(system='http://hl7.org/fhir/observation-category',code='vital-signs',display='Vital Signs')]}],code={'text':'BP'},effectiveDateTime=at,subject=subject,valueString=re.search(r'"([^"]+mmHg)"',q).group(1)))
    elif n in (4,7):
        answer=[last if measures else -1]
    elif n==6:
        answer=[sum(m[1] for m in measures)/len(measures) if measures else -1]
    elif n==5 and last is not None and last<=1.9:
        dose=4 if last<1 else 2 if last<1.5 else 1
        posts.append(medication(re.search(r'\d{4}-\d{4}-\d{2}',ctx).group(),'IV',dict(doseQuantity=dict(value=dose,unit='g'),rateQuantity=dict(value=dose,unit='h'))))
    elif n==8:
        body=service('http://snomed.info/sct',re.search(r'\d{9,}',ctx).group())
        body['note']={'text':re.search(r'"([^"]+)"',q).group(1)}
        posts.append(body)
    elif n==9 and last is not None and last<3.5:
        posts.append(medication(re.search(r'\d{5}-\d{3}-\d{2}',ctx).group(),'oral',dict(doseQuantity=dict(value=round((3.5-last)*100,3),unit='mEq'))))
        body=service('http://loinc.org','2823-3')
        body['occurrenceDateTime']='2023-11-14T08:00:00+00:00'
        posts.append(body)
    elif n==10:
        answer=[last,measures[0][2]] if measures else [-1]
        if not measures or measures[0][0]<MONITOR.replace(year=MONITOR.year-1):
            posts.append(service('http://loinc.org','4548-4'))
    hist=[]
    for post in posts:
        hist.extend([('agent','POST '+p.BASE+post['resourceType']+'\n'+json.dumps(post)),('user','POST request accepted and executed successfully.')])
    hist.append(('agent','FINISH('+json.dumps(answer)+')'))
    passed=p.grade(case,hist,json.dumps(answer),reads)[1]
    if return_trace:
        return passed,rows,posts,answer
    return passed,bool(posts)


def environment(cases):
    counts=collections.defaultdict(collections.Counter)
    for c in cases:
        good,writes=canonical(c)
        key=p.family(c)
        counts[key]['n']+=1
        counts[key]['pass']+=good
        if kind(c) in (5,9,10):
            counts[key]['write' if writes else 'no_write']+=1
    # Missing branches exercise the identical simulator and official scoring logic
    # on a synthetic lab bundle, without touching the prepared chart database.
    synthetic={}
    for n in (5,9,10):
        for branch in ('write','no_write'):
            if counts['task'+str(n)][branch]: continue
            c=next(c for c in cases if kind(c)==n)
            value={5:1.2 if branch=='write' else 2.2,9:3.1 if branch=='write' else 4.1,10:6.2}[n]
            when=MONITOR.replace(year=2021) if n==10 and branch=='write' else MONITOR
            item=dict(resourceType='Observation',effectiveDateTime=when.isoformat(),valueQuantity={'value':value})
            raw={'entry':[{'resource':item}]}
            original_fetch=globals()['fetch']
            original_read=p.ref.send_get_request
            try:
                globals()['fetch']=lambda u:(json.dumps({'rows':[item]}),('Observation',parse_qs(urlsplit(u).query)))
                p.ref.send_get_request=lambda u:{'data':json.dumps(raw),'status_code':200}
                passed,posted=canonical(c)
                synthetic[str(n)+'_'+branch]=passed and posted==(branch=='write')
            finally:
                globals()['fetch']=original_fetch
                p.ref.send_get_request=original_read
    p.save('controls.json',dict(real=counts,synthetic=synthetic))
    assert all(v['pass']==v['n'] for v in counts.values()) and all(synthetic.values()),counts


def load_cases():
    labels=json.loads((PRIOR/'split.json').read_text())['discovery']
    all_cases={c['id']:c for c in json.loads((p.DATA/'test_data_v2.json').read_text())}
    return [all_cases[i] for i in labels]


def report(cases):
    arms={k:json.loads(Path(k+'.json').read_text()) for k in ('plain','old','guide','assisted')}
    base={r['id']:r for r in arms['plain']}
    out={}
    for arm,rows in arms.items():
        group={}
        for n in [0,*range(1,11),'focus']:
            selected=[r for r in rows if n==0 or kind(r)==n or (n=='focus' and kind(r) in (2,4,6))]
            if not selected:continue
            ctr=collections.Counter()
            for r in selected:
                b=base[r['id']]
                c=cases[r['id']]
                read=False
                if kind(c) not in WRITES:
                    expected=urlsplit(query(c))
                    required=parse_qs(expected.query)
                    for (role,t),(_,reply) in zip(r['history'],r['history'][1:]):
                        if role!='agent' or not t.startswith('GET ') or not reply.startswith('{"returned":'):continue
                        u=urlsplit(t[4:]);q=parse_qs(u.query)
                        if u.path==expected.path and all(set(v)<=set(q.get(k,[])) for k,v in required.items()):read=True
                ctr.update(n=1,qualified=int(r['qualified']),official=int(r['official']),gains=int(r['qualified'] and not b['qualified']),losses=int(b['qualified'] and not r['qualified']),canonical_read=int(read),strict=int(r['qualified'] and (read or kind(c) in WRITES)),any_post=int(any(role=='agent' and t.startswith('POST ') for role,t in r['history'])))
            group[str(n)]=ctr
        out[arm]=group
    p.save('results.json',out)


if __name__=='__main__':
    cases=load_cases()
    arm=os.environ['ARM']
    p.ref=p.scorer()
    if arm=='controls':
        environment(cases)
    elif arm=='summary':
        report({c['id']:c for c in cases})
    else:
        p.fetch=model_fetch
        p.template=next(ast.literal_eval(n.value) for n in ast.parse((p.SOURCE/'__init__.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='MedAgentBench_prompt')
        p.docs=json.dumps(DOCS).replace('{api_base}',p.BASE.rstrip('/'))
        p.tok=p.AutoTokenizer.from_pretrained(p.MODEL,padding_side='left',local_files_only=True)
        p.model=p.Qwen2_5_VLForConditionalGeneration.from_pretrained(p.MODEL,torch_dtype=p.torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True).eval()
        if os.getenv('SINGLE_DECODE')=='1':
            # During generation only the last input position predicts the next token.
            p.model.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
        bank=json.loads((PRIOR/'bank.json').read_text())
        hints={'plain':None,'old':lambda c:p.nearest(c['instruction'],bank)['hint'],'assisted':None,'guide':lambda c:GUIDES[kind(c)].replace('<base>',p.BASE.rstrip('/'))}
        assisted=arm=='assisted'
        p.run([c for c in cases if not assisted or kind(c) not in WRITES],arm,hints[arm],initial_query=query if assisted else None,batch_size=1 if os.getenv('SINGLE_DECODE')=='1' else 4)
