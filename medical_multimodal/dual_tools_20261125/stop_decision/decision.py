import json
from pathlib import Path


def read(root,file):
    return json.loads((Path(root)/file).read_text())


def visible(q,notes):
    return 'Question: '+q+'\nObserved:\n'+('\n'.join(n['name']+': '+n['text'][:450] for n in notes) or '(none)')


def prompt(q,notes,trusted,answer,accepted,planned,skill,hint):
    name=notes[-1]['name']
    state=('Question: '+q+'\nAccepted evidence: '+visible(q,trusted)+'\nCurrent answer: '+answer
           +'\nLast result was '+('accepted' if accepted else 'discarded')+' from '+name+
           '. Tools already tried: '+', '.join(v['name'] for v in notes)+'. Remaining calls: '+str(4-len(notes))+
           '. Router proposes: '+planned)
    return state,skill+'\n'+hint


def ballot(choose,state,rule,stop_letter,first_letter='A',opened=None):
    meanings={stop_letter:'stop and submit current answer',('B' if stop_letter=='A' else 'A'):'continue acquiring evidence'}
    order=first_letter+('B' if first_letter=='A' else 'A')
    instruction=rule+'\n'+' '.join(f'{k} = {meanings[k]}.' for k in order)+' Select one.'
    return choose(state,opened or {},instruction,list('AB'))==stop_letter


def moves(a,b):
    return dict(n=len(a),before=sum(a),after=sum(b),gain=sum(x<y for x,y in zip(a,b)),loss=sum(x>y for x,y in zip(a,b)))
