"""Cross the decision context and the action presentation on the 23 frozen first-call states."""
import json,os
from pathlib import Path
from PIL import Image
from clinical import DATA,NAMES,data,split
from decision import ballot,prompt,read,visible
from feedback import support
from med import SEED,MENU,CODES
from shared import models,save


def run():
    cases,_=data();_,_,ids=split()
    base=read(os.environ['PAIRED_RUN'],'private_test.json')
    old=read(os.environ['BOUNDED_RUN'],'private_test.json')
    skill=read(os.environ['BOUNDED_RUN'],'private_skill.json')['stop']
    file=Path(os.environ['OUTCOME_FIT_CACHE'])/'memory/raw.jsonl'
    bank=[json.loads(json.loads(x)['question']) for x in file.read_text().splitlines()]
    choose,_=models()
    rows={}
    for k in ids:
        notes=base[k]['baseline']['notes']
        if len(notes)<2:continue
        first=notes[0]; next_name=notes[1]['name'];q=cases[k]['question']
        e=old[k]['learned']['events'][0];accepted=e['accepted']
        state,rule=prompt(q,[first],[first] if accepted else [],old[k]['learned']['answer'],accepted,next_name,skill,
                          support(bank,first['name'],q,first['text'][:450]))
        opts=[CODES[n] for n in NAMES if not(first['ok'] and n==first['name'] and n in ('OCR','ImageDescription'))]+['E']
        options=MENU+'\nAvailable letters: '+' '.join(opts)
        with Image.open(DATA/cases[k]['images'][0]) as source:img=source.convert('RGB')
        axes={}
        for visual,media in [('text',{}),('image',{'image':img})]:
            for history,ctx in [('stop_state',state),('observed_only',visible(q,[first]))]:
                for guide,guidance in [('skill',rule),('seed',SEED)]:
                    # Only the action prompt/allowed tokens change between the paired cells.
                    key='_'.join([visual,history,guide])
                    axes[key+'_binary']=ballot(choose,ctx,guidance,'A',opened=media)
                    action=choose(ctx,media,guidance+'\n'+options,opts)
                    axes[key+'_menu']=action=='E'
                    if (visual,history,guide)==('image','observed_only','seed'):
                        axes['base_action_reproduced']=(action==CODES[next_name])
        rows[k]=axes
    assert len(rows)==23
    save('private_framing',rows)
    save('framing_result',{k:sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))})


if __name__=='__main__':run()
