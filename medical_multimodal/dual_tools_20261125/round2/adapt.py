import json
from shared import ROOT, save
from previous_learning import update_memory, words
from upstream_skill import apply_patch_with_report


def consolidate(examples, gen):
    d=ROOT/'memory'
    d.mkdir()
    return update_memory({'memories':[]},'fit',examples,d,gen)[0]


def append(seed, instruction):
    updated, receipt=apply_patch_with_report(seed,{'edits':[{'op':'append','content':instruction}]})
    if receipt[0]['status']!='applied_append' or updated==seed:
        raise RuntimeError('native skill update was not applied')
    return updated


def retrieval(query, stage, bank):
    pool=[x for x in bank if x['stage']==stage] or bank
    q=words(query)
    match=max(pool,key=lambda x:len(q & words(x['visible']))/max(1,len(q | words(x['visible']))))
    return match['visible'][:360]+'\nTraining next action: '+match['next']


def metrics(rows):
    return {key:sum(x[key] for x in rows)/len(rows) for key in rows[0]}
