"""Pinned formula library. No benchmark cases or answer values are imported here."""
import inspect,json,math,re,sys,types,typing,unicodedata
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/medical_calc'))
from src.domain.services.calculators import CALCULATORS
CALCS={c().tool_id:c() for c in CALCULATORS}
# Explicit identity aliases, reviewed from public names and learning-only diagnosis.
# Never equate NEWS with NEWS2, SIRS with sepsis, or Wells PE with Wells DVT.
ALIASES={
 'shock_index':['Shock Index','Heart Rate/Systolic Blood Pressure Ratio'],
 'wells_pe':["Wells' Criteria for Pulmonary Embolism",'Wells Score for Pulmonary Embolism'],
 'wells_dvt':['Wells Criteria for Deep Vein Thrombosis'],
 'news2_score':['National Early Warning Score (NEWS) 2','National Early Warning Score 2','NEWS2'],
 'glasgow_coma_scale':['GCS'],
 'qsofa_score':['qSOFA (Quick SOFA) Score for Sepsis','qSOFA'],
 'sofa_score':['Sequential Organ Failure Assessment (SOFA) Score'],
 'stop_bang':['STOP-BANG Score for Obstructive Sleep Apnea'],
 'four_at':['4 A’s Test for Delirium Screening','4AT'],
 'nrs_2002':['Nutrition Risk Screening 2002 (NRS-2002)'],
 'braden_scale':['Braden Score for Pressure Ulcers'],
 'rts':['Revised Trauma Score'],
 'triss':['Trauma Score and Injury Severity Score (TRISS)'],
 'psi_port':['PSI/PORT Score: Pneumonia Severity Index for CAP'],
 'perc_rule':['PERC Rule for Pulmonary Embolism'],
 'spesi':['Simplified PESI (Pulmonary Embolism Severity Index)'],
 'mascc_score':['MASCC Risk Index for Febrile Neutropenia'],
 'gad7':['GAD-7 (General Anxiety Disorder-7)'],
 'hama':['Hamilton Anxiety Scale']}
def norm(s):
    s=unicodedata.normalize('NFKC',str(s)).casefold().replace('’',"'")
    return ' '.join(re.findall(r'[a-z0-9]+',s))
NAMES={}
for cid,c in CALCS.items():
    for n in [cid,c.name,*ALIASES.get(cid,[])]:
        key=norm(n)
        if key in NAMES and NAMES[key]!=cid:raise RuntimeError('ambiguous calculator alias')
        NAMES[key]=cid
def identity(name):return NAMES.get(norm(name),'unregistered:'+norm(name))
def typed_signature(c):
    try:
        hints=typing.get_type_hints(c.calculate)
        for name in inspect.signature(c.calculate).parameters:type_schema(hints[name])
        return True
    except (TypeError,KeyError):return False

@lru_cache(maxsize=1)
def coverage():
    d=json.loads((ROOT/'formula_coverage.json').read_text())
    if d['exit_code']!=0:raise RuntimeError('formula validation failed')
    return {k:v for k,v in d['tested_calculators'].items() if typed_signature(CALCS[k])}
def type_schema(t):
    origin=typing.get_origin(t);args=typing.get_args(t)
    if origin in (typing.Union,types.UnionType):return {'anyOf':[type_schema(x) for x in args]}
    if origin is typing.Literal:return {'enum':list(args)}
    if origin in (list,tuple):return {'type':'array','items':type_schema(args[0])}
    if t in (int,float,bool,str,type(None)):return {'type':{int:'integer',float:'number',bool:'boolean',str:'string',type(None):'null'}[t]}
    raise TypeError('unsupported calculator type '+str(t))
# Source-documented categorical domain. Do not normalize guessed labels.
REVIEWED_ENUMS={'mascc_score':{'burden_of_illness':['none_mild','moderate','severe']}}
OPTIONAL_REVIEWED={'shock_index':{'diastolic_bp'},'spesi':{'heart_rate','systolic_bp','spo2','heart_rate_gte_110','sbp_lt_100','spo2_lt_90'},'sofa_score':{'urine_output_24h'}}
@lru_cache(maxsize=160)
def details(cid):
    if cid not in CALCS:raise ValueError('unknown calculator_id')
    c=CALCS[cid];sig=inspect.signature(c.calculate);hints=typing.get_type_hints(c.calculate)
    properties={};required=[]
    for name,p in sig.parameters.items():
        try:properties[name]=type_schema(hints[name])
        except TypeError:properties[name]={'description':'Unresolved type; this calculator is not executable in this adapter.'}
        # Force explicit clinical inputs; upstream convenience defaults cannot silently invent facts.
        # A nullable default means this input may be absent. Clinical false/zero
        # defaults remain explicit so absence is never silently treated as negative.
        if not (p.default is None and name in OPTIONAL_REVIEWED.get(cid,set())):required.append(name)
        if p.default is not inspect.Parameter.empty:
            properties[name]['upstream_default_not_assumed']=p.default
            if p.default is None and name in OPTIONAL_REVIEWED.get(cid,set()):properties[name]['description']='Reviewed optional/alternative input; see conditional requirements. Missing does not establish a negative clinical finding.'
    for name,values in REVIEWED_ENUMS.get(cid,{}).items():
        properties[name]['enum']=list(values)
    return {'calculator_id':cid,'name':c.name,'aliases':ALIASES.get(cid,[]),
      'purpose':c.metadata.low_level.purpose,'execution_enabled':cid in coverage(),
      'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False},
      'instructions':inspect.getdoc(c.calculate),'conditional_requirements':(['Supply a raw measurement or its explicit threshold flag for each of heart rate, systolic BP and oxygen saturation.'] if cid=='spesi' else []),
      'references':[x.to_dict() for x in c.references],
      'validation':'Pinned upstream regression tests; independent sentinel checks separately recorded. Research use; not clinical certification.'}
def search(query='',offset=0):
    if type(query)!=str or len(query)>200 or type(offset)!=int or offset<0:raise ValueError('invalid query or offset')
    terms=set(norm(query).split());rows=[];tested=coverage()
    for cid,c in CALCS.items():
        words=set(norm(' '.join([cid,c.name,c.metadata.low_level.purpose,*map(str,c.metadata.high_level.conditions),*map(str,c.metadata.high_level.keywords),*ALIASES.get(cid,[])])).split())
        hit=len(terms&words)
        if norm(query) in {norm(x) for x in [cid,c.name,*ALIASES.get(cid,[])]}:hit+=1000
        if not terms or hit:rows.append((hit,{'calculator_id':cid,'name':c.name,'purpose':c.metadata.low_level.purpose,'execution_enabled':cid in tested}))
    rows.sort(key=lambda r:(-r[0],r[1]['calculator_id']))
    return {'calculators':[x[1] for x in rows[offset:offset+15]],'total':len(rows),'next_offset':offset+15 if offset+15<len(rows) else None,'note':'Directory is shared across tasks; relevance is lexical, not an oracle. Inspect exact definition and units before selecting.'}
def validate_conditional_inputs(cid,params):
    # Nullable signature parameters are not permission to assume negative findings.
    if cid=='spesi':
        for raw,flag in [('heart_rate','heart_rate_gte_110'),('systolic_bp','sbp_lt_100'),('spo2','spo2_lt_90')]:
            if params.get(raw) is None and params.get(flag) is None:
                raise ValueError('one of '+raw+' or '+flag+' must be explicitly supplied')

def execute(cid,params):
    from jsonschema import Draft202012Validator
    d=details(cid)
    if not d['execution_enabled']:raise ValueError('calculator has no passing execution test; not enabled')
    from jsonschema import ValidationError
    try:Draft202012Validator(d['parameters']).validate(params)
    except ValidationError as e:raise ValueError('Invalid calculator parameters: '+e.message[:300]) from e
    def finite(v):
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite input')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        if isinstance(v,list):
            for x in v:finite(x)
    finite(params)
    validate_conditional_inputs(cid,params)
    result=CALCS[cid].calculate(**params).to_dict();finite(result)
    return {'calculator_id':cid,'name':CALCS[cid].name,'value':result['value'],'unit':result['unit'],
      'calculation_details':result['calculation_details'],'interpretation':result['interpretation'],
      'formula_used':result['formula_used'],'references':d['references'],'input_warnings':result['input_warnings']}
