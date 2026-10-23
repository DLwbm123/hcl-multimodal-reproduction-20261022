"""Explicit calculator identity plus episode matching. No token-overlap scoring."""
import math,re,unicodedata
from clinical_catalog import identity,coverage

def value_match(pred,gold):
    if pred is None or isinstance(pred,bool):return False
    pattern=r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z%/²0-9]*)\s*$'
    def numeric(x):
        if type(x) in (int,float):return (float(x),'') if math.isfinite(x) else None
        m=re.fullmatch(pattern,str(x))
        if m and math.isfinite(float(m[1])):return float(m[1]),m[2].casefold()
    g=numeric(gold)
    if g:
        p=numeric(pred)
        return bool(p and (not p[1] or p[1]==g[1]) and abs(p[0]-g[0])<=max(.1,abs(g[0])*.02))
    def category(x):
        s=unicodedata.normalize('NFKC',str(x)).strip().casefold()
        s=re.sub(r'^g(?=\d)','',s);s=re.sub(r'^(grade|class|stage)\s*','',s)
        return re.sub(r'\s+',' ',s)
    return category(pred)==category(gold)

def score_prediction(pred,task):
    from medharness.scorers.calc import answer_episode_ids
    episodes=answer_episode_ids(task);answers=task['calculator_answers']
    predictions=pred.get('calculators',[]) if isinstance(pred,dict) else []
    P={identity(c.get('name','')) for c in predictions if c.get('name')};G={identity(c['name']) for c in answers}
    matched=len(P&G);prec=matched/len(P) if P else 0.;rec=matched/len(G) if G else 0.;f1=2*prec*rec/(prec+rec) if prec+rec else 0.
    keys={}
    for c in predictions:
        k=(identity(c.get('name','')),str(c.get('episode_id','')));keys.setdefault(k,[]).append(c)
    rows=[]
    for a,e in zip(answers,episodes):
        matches=keys.get((identity(a['name']),str(e)),[])
        # Duplicate identity/episode is ambiguous: do not choose the most favorable value.
        selected=len(matches)==1;v=matches[0].get('value') if selected else None
        rows.append({'calc':a['name'],'gold':a['final_answer'],'pred':v,'selected':selected,'ok':selected and value_match(v,a['final_answer']),
          'kind':'correct' if selected and value_match(v,a['final_answer']) else ('wrong_value_or_abstention' if selected else 'missing_or_ambiguous_identity_episode')})
    correct=sum(r['ok'] for r in rows);n=len(rows);selected=sum(r['selected'] for r in rows)
    return {'selection':{'precision':prec,'recall':rec,'f1':f1,'matched':matched,'gold':len(G),'predicted':len(P)},
      'n_correct':correct,'n_scored':selected,'n_abstained':sum(r['pred'] is None for r in rows),'rows':rows,
      'value_accuracy':correct/n if n else 0.,'strict_value_accuracy':correct/n if n else 0.,'utility':f1*correct/n if n else 0.,
      'diagnostic':{'matched_outputs':selected,'unmatched_outputs':n-selected,'wrong_values_after_match':selected-correct,
      'conditional_value_accuracy':correct/selected if selected else 0.,'extra_calculator_types':len(P-G),
      'formula_covered_outputs':sum(identity(a['name']) in coverage() for a in answers)}}
