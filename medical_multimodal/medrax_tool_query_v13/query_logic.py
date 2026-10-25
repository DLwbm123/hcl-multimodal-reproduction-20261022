"""Restricted persistent query program; never executes generated code."""
import json,re
ARMS=['H0_static','fixed_query','learned_query']
TOOLS=['chest_xray_expert','llava_med_qa']
FINAL='If you had to choose the best option, only respond with the letter of choice (only one of A, B, C, D, E, F)'
FIXED_QUERY='The following question is context for examining this single image: {question_stem}\nDescribe only findings directly visible in THIS image, including anatomical location, side, extent and measurable features when an adequate scale is present. State what cannot be determined. Do not compare with an image you were not given, infer chronology or an unprovided clinical outcome, select an answer option, or invent a diagnosis. Provide observations useful for the stated question; distinguish observations from uncertain interpretations.'


def stem(question):
    m=re.search(r'(?m)^\s*[A-F][.)]\s+',question)
    return (question[:m.start()] if m else question).strip()[:2000]


def validate(value):
    assert set(value)=={'admit','tool_name','query_template','source_indices','rationale'}
    assert type(value['admit']) is bool and isinstance(value['rationale'],str) and 0<len(value['rationale'])<=1200
    if not value['admit']:return value
    assert value['tool_name'] in TOOLS and isinstance(value['query_template'],str) and 250<=len(value['query_template'])<=1600
    text=value['query_template'];assert text.count('{question_stem}')==1 and '{' not in text.replace('{question_stem}','') and '}' not in text.replace('{question_stem}','')
    assert isinstance(value['source_indices'],list) and 4<=len(value['source_indices'])<=24 and all(type(i) is int and 0<=i<24 for i in value['source_indices']) and len(set(value['source_indices']))==len(value['source_indices'])
    assert text!=FIXED_QUERY
    return value


def parse(text):
    s=text.strip()
    if s.startswith('```'):s=re.sub(r'^```(?:json)?\s*|\s*```$','',s)
    return validate(json.loads(s))


def query(policy,question):return policy['query_template'].replace('{question_stem}',stem(question))


def arguments(tool_name,image_path,text):
    assert tool_name in TOOLS
    return {'image_paths':[image_path],'prompt':text} if tool_name=='chest_xray_expert' else {'image_path':image_path,'question':text}


def prefix_text(rows):
    if not rows:return ''
    # Query templates, source feedback and arm identity are not sent to the controller.
    values=[{'current_image_index':r['image_index'],'tool':r['tool_name'],'status':r['status'],'observation':r['observation']} for r in rows]
    return '\nExternal tool observations actually executed on the CURRENT CASE, one image at a time. Treat these as fallible observations, not verified diagnoses or evidence from another patient. They cannot establish chronology or facts absent from the original input. Reconcile with the original images and question; retain independent judgement.\n'+json.dumps(values,ensure_ascii=False)


def decision(pairs,by,ids,gold,complete):
    acquired=[]
    for k in ids:
        ok=True
        for b in [1,2]:
            h=by['H0_static_r'+str(b)][k];c=by['fixed_query_r'+str(b)][k];m=by['learned_query_r'+str(b)][k]
            ok=ok and h.get('predicted')!=gold[k] and m.get('predicted')==gold[k] and bool(m.get('prefix_text')) and m.get('prefix_text')!=c.get('prefix_text')
        if ok:acquired.append(k)
    passed=complete and len(acquired)>=2 and all(p['H0_to_learned']['gain']-p['H0_to_learned']['loss']>=2 and p['H0_to_learned']['loss']<=1 and p['fixed_to_learned']['gain']-p['fixed_to_learned']['loss']>=1 for p in pairs)
    return {'pass':bool(passed),'repeat_observed_gain_with_different_observations_count':len(acquired),'repeat_observed_gain_ids':acquired,'exploratory_error_enriched_benchmark_only':True}
