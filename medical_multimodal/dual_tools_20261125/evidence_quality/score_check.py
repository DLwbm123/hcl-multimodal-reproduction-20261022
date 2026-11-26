"""Evaluation-only question-context sensitivity with unchanged grader and fixed predictions."""
import json,os,time
from pathlib import Path
from clinical import data,split
from decision import read,visible,moves
from shared import models,save


def run():
    choose,gen=models();cases,refs=data();_,gate,test=split()
    src=Path(os.environ['VETO_RUN']);parent=Path(os.environ['QUALITY_RUN'])
    previous=Path(os.environ['PAIRED_RUN']);audit=read(os.environ['AUDIT_RUN'],'private_audit.json')
    output={}
    for phase,ids in [('gate',gate),('test',test)]:
        old=read(src,'private_'+phase+'.json');base=read(previous,'private_'+phase+'.json')
        first={}
        def grade(k,answer):
            return int(choose('Question: '+cases[k]['question']+'\nReference acceptance criteria: '+json.dumps(refs[k]['gt_answer'])+'\nAnswer: '+answer,{},
                    'A = answer satisfies the reference clinically including key details, B = incorrect or incomplete. Return A or B.',list('AB'))=='A')
        for k in ids:
            r=old[k];notes=base[k]['baseline']['notes'][:r['steps']]
            accepted=[n for n,e in zip(notes,r['events']) if e['accepted']]
            text=(audit[k]['original'] if k in audit else gen(visible(cases[k]['question'],accepted)+'\nGive the concise clinical answer supported by this evidence.',96))
            first[k]=dict(old_answer=text,old=grade(k,text),base=grade(k,base[k]['baseline']['answer']))
            if k in audit:first[k]['reference_control']=grade(k,str(refs[k]['gt_answer']['whitelist'][0][0]))
        while not (parent/('private_'+phase+'.json')).is_file():
            if (parent/'SUPERVISOR_EXIT.json').is_file():raise RuntimeError('revised run exited before predictions')
            time.sleep(3)
        new=read(parent,'private_'+phase+'.json')
        for k in ids:first[k]['new']=grade(k,new[k]['answer'])
        save('private_'+phase,first)
        groups={'all':ids}
        if phase=='test':groups['old_errors']=list(audit)
        output[phase]={}
        for name,keys in groups.items():
            output[phase][name]=dict(n=len(keys),base=sum(first[k]['base'] for k in keys),
                old=sum(first[k]['old'] for k in keys),new=sum(first[k]['new'] for k in keys),
                reference_pass=sum(first[k].get('reference_control',0) for k in keys),
                revisions=moves([first[k]['old'] for k in keys],[first[k]['new'] for k in keys]))
        save('score_context_result',output)


if __name__=='__main__':run()
