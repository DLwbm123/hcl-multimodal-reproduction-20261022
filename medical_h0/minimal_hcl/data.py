"""Predeclared synthetic arithmetic tasks; no benchmark or patient records."""
import json, random
from pathlib import Path
from tools import COLUMNS, calculate

# Each role has disjoint instances, language/order/distractors and recipe variants.
# Shared arithmetic families are intentional: this is instance transfer, not a claim
# of unseen formula-family or clinical generalization.
SPECS={
 'train_dependency':[(12,3,4,'n/d','r*s'),(18,6,7,'n-d','r+s'),(25,5,2,'n/d','r-s'),(9,4,3,'n+d','r/s')],
 'select_dependency':[(27,9,6,'n/d','r+s'),(13,5,4,'n-d','r*s'),(8,7,3,'n+d','r-s'),(32,4,2,'n/d','r**s')],
 'anchor_dependency':[(22,11,5,'n/d','r*s'),(31,7,8,'n-d','r/s'),(6,3,4,'n+d','r+s'),(19,4,2,'n-d','r-s')],
 'confirm_dependency':[(14,7,3,'n/d','(r+s)/s'),(21,4,2,'n-d','r*s+s'),(11,6,5,'n+d','(r-s)/s'),(36,6,3,'n/d','r/s+s')],
 'train_submission':[(45,5,2,'n/d','r*s'),(70,1.75,0,'n/(d**2)',None),(16,4,3,'n-d','r+s'),(24,3,0,'n/d',None)],
 'select_submission':[(63,7,4,'n/d','r-s'),(81,1.8,0,'n/(d**2)',None),(17,8,2,'n+d','r/s'),(30,6,0,'n-d',None)],
 'confirm_submission':[(50,10,3,'n/d','r+s*s'),(72,1.6,0,'n/(d**2)',None),(29,5,4,'n-d','r/s'),(42,7,0,'n+d',None)],
 'anchor_basic':[(68,1.7,0,'n/(d**2)',None),(18,3,0,'n/d',None),(11,8,0,'n+d',None),(27,9,0,'n-d',None)],
 'confirm_basic':[(98,1.4,0,'n/(d**2)',None),(28,4,0,'n/d',None),(23,6,0,'n-d',None),(7,9,0,'n+d',None)]}

def build():
    out=[]
    for role,specs in SPECS.items():
        for i,(n,d,s,first,second) in enumerate(specs):
            cid=f'{role}_{i}';patient=f'SYN-{list(SPECS).index(role)*4+i:03d}'
            names=['Synthetic Primary Quantity','Synthetic Derived Quantity']
            if 'submission' in role:names=[['Synthetic Reported Quotient','Body Mass Index (BMI)','Synthetic Adjusted Quantity','Synthetic Difference Quantity'][i], 'Synthetic Followup Quantity']
            elif first=='n/(d**2)':names[0]='Body Mass Index (BMI)'
            # Names are output labels, not fixed clinical formulas (except explicitly defined BMI).
            definitions=f'{names[0]} = {first}.'
            if second:definitions+=f' {names[1]} = {second}, where r denotes {names[0]}.'
            instruction=[f'For {patient} in episode_000, return these outputs: {definitions}',
                         f'The requested report is for episode_000 of {patient}. Recipes: {definitions}',
                         f'Use only the current serum/chart observations of {patient}, episode_000. Required quantities: {definitions}',
                         f'Produce a calculator report for {patient} (episode_000). Definitions: {definitions}'][i]
            prompt=instruction+' n, d and s are field names in the facts. Read the facts and use calculate. Submit the requested named quantities. Other patients, specimens and episodes are distractors. These are synthetic exercises with supplied recipes, not clinical recommendations.'
            vals=[(patient,'n',n,'episode_000','number','Serum'),(patient,'d',d,'episode_000','number','Serum'),(patient,'s',s,'episode_000','number','Chart'),('DISTRACTOR','n',901+i,'episode_000','number','Serum'),(patient,'n',100+n,'episode_old','number','Urine')]
            random.Random(20261023+len(out)).shuffle(vals)
            facts=[dict(zip(COLUMNS,(p,f,str(v),'2026-01-01',e,u,t,'synthetic'))) for p,f,v,e,u,t in vals]
            r=calculate(first,{'n':n,'d':d,'s':s})
            answers=[{'name':names[0],'final_answer':r,'inputs':[]}]
            actions=[('read_facts',{}),('calculate',{'expression':first,'variables':{'n':n,'d':d,'s':s}})]
            if second:
                y=calculate(second,{'r':r,'s':s})
                answers.append({'name':names[1],'final_answer':y,'inputs':[{'field':names[0],'value':r}]})
                actions.append(('calculate',{'expression':second,'variables':{'r':r,'s':s}}))
            actions.append(('submit_answer',{'calculators':[{'name':a['name'],'episode_id':'episode_000','value':a['final_answer']} for a in answers]}))
            out.append({'id':cid,'role':role,'prompt':prompt,'rows':facts,'gold':{'calculator_answers':answers,'required_calculators':[a['name'] for a in answers]},'script':actions})
    return out

def freeze(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cases=build();assert len(cases)==36 and len({c['id'] for c in cases})==36
    visible=[{k:c[k] for k in ('id','role','prompt','rows')} for c in cases]
    (root/'visible_split.json').write_text(json.dumps(visible,indent=2))
    (root/'evaluator_gold.json').write_text(json.dumps({c['id']:c['gold'] for c in cases},indent=2))
    (root/'scripted_actions.json').write_text(json.dumps({c['id']:c['script'] for c in cases},indent=2))
if __name__=='__main__':
    import sys
    freeze(sys.argv[1])
