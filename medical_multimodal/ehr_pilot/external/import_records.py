"""Populate only a fresh loopback HAPI instance; no model output is consumed."""
import json
from pathlib import Path
import requests
from urllib.parse import urlsplit

base='http://127.0.0.1:18478/fhir/'
session=requests.Session()
session.trust_env=False
records=json.loads(Path('resources.json').read_text())
# Stable PUT URLs allow a failed transaction to be retried without duplicate measurements.
for kind in ['Patient','Observation']:
    subset=[r for r in records if r['resourceType']==kind]
    for start in range(0,len(subset),60):
        entries=[]
        for i,r in enumerate(subset[start:start+60], start):
            r.setdefault('id', 'O'+str(i+1))
            entries.append(dict(resource=r,request={'method':'PUT','url':kind+'/'+r['id']}))
        response=session.post(base,json=dict(resourceType='Bundle',type='transaction',entry=entries),timeout=120)
        if response.status_code!=200: raise RuntimeError((response.status_code,response.text[:650]))
        assert all(e['response']['status'].startswith(('200','201')) for e in response.json()['entry'])
    res=session.get(base+kind,params={'_summary':'count'},timeout=30)
    res.raise_for_status()
    assert res.json()['total']==len(subset)
    print(kind,len(subset),flush=True)
