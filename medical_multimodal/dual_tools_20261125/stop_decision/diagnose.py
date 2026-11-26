import os,json
from collections import Counter
from pathlib import Path
from PIL import Image
from clinical import DATA,data,split
from decision import read,visible,prompt,ballot,moves
from feedback import support
from shared import models,save


def run():
    cases,refs=data(); _,_,ids=split()
    base=read(os.environ['PAIRED_RUN'],'private_test.json')
    root=Path(os.environ['OUTCOME_FIT_CACHE'])
    bank=[json.loads(r['question']) for r in (json.loads(x) for x in (root/'memory/raw.jsonl').read_text().splitlines())]
    choose,gen=models()
    arms={k:(read(v,'private_test.json'),read(v,'private_skill.json')['stop']) for k,v in
          [('bounded',os.environ['BOUNDED_RUN']),('free',root)]}
    rows=[]
    for i,k in enumerate(ids,1):
        q=cases[k]['question'];b=base[k]['baseline'];notes=b['notes'];first=notes[0];focus=len(notes)>1
        planned=notes[1]['name'] if focus else 'finish'
        with Image.open(DATA/cases[k]['images'][0]) as src: img=src.convert('RGB')
        row=dict(focus=focus,first=first['name'],next=planned,baseline=base[k]['base_score']['correct'],horizon=len(notes))
        hint=support(bank,first['name'],q,first['text'][:450])
        for arm,(source,skill) in arms.items():
            saved=source[k]['learned'];ev,=saved['events']
            assert ev['tool']==first['name'] and ev['stop'] and ev['base_stop']==(not focus)
            trusted=[first] if ev['accepted'] else []
            state,rule=prompt(q,[first],trusted,saved['answer'],ev['accepted'],planned,skill,hint)
            votes={}
            for label,letter,position in [('original','A','A'),('swap','B','A'),('reverse_lines','A','B'),('swap_reverse','B','B')]:
                votes[label]=ballot(choose,state,rule,letter,position)
            if arm=='bounded':
                votes['with_image']=ballot(choose,state,rule,'A',opened={'image':img})
                votes['no_memory']=ballot(choose,state,skill+'\n','A')
            row[arm]=dict(accepted=ev['accepted'],correct=saved['correct'],**votes)
        if focus:
            grades=[]
            for n in range(1,len(notes)+1):
                if n==len(notes):
                    score=row['baseline']
                else:
                    answer=gen(visible(q,notes[:n])+'\nGive the concise clinical answer supported by this evidence.',96)
                    score=int(choose('Reference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+answer,{},
                                 'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))=='A')
                grades.append(score)
            row['grades']=grades
        rows.append(row)
        if i%5==0:print('replay',i,flush=True)
    save('private_replay',dict(zip(ids,rows)))
    report={}
    for name,select in [('all',rows),('disagreed',[r for r in rows if r['focus']]),('already_stop',[r for r in rows if not r['focus']])]:
        report[name]={'n':len(select)}
        for arm in arms:
            report[name][arm]={key:sum(r[arm][key] for r in select) for key in select[0][arm]}
        report[name]['next']=dict(Counter(r['next'] for r in select))
    f=[r for r in rows if r['focus']]
    report['baseline_prefix_first_second']=moves([r['grades'][0] for r in f],[r['grades'][1] for r in f])
    report['baseline_prefix_first_final']=moves([r['grades'][0] for r in f],[r['grades'][-1] for r in f])
    report['old_bounded_to_baseline']=moves([r['bounded']['correct'] for r in f],[r['baseline'] for r in f])
    save('replay_result',report)


if __name__=='__main__':run()
