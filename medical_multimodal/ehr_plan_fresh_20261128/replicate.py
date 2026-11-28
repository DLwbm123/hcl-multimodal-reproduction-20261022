"""Train six adapters, lock their deployment-ranked selections, then evaluate once."""
import json
import os
from pathlib import Path
from statistics import mean, stdev
import time

import torch
from peft.utils.save_and_load import load_peft_weights

from panel import FORMS, view
from run import STATES
from train import Policy, load_split, save
import r0
import r1

SEEDS = (26112862, 26112863, 26112864)
ARMS = ('a', 'b')


def rank(metrics):
    c=metrics['counts']
    extra=c['extra_reads'] if 'extra_reads' in c else metrics['guard']['extra_reads']
    return c['guarded_success'], -(c['tool_calls']+extra)


def run_name(arm, seed):
    return Path(f'{arm}{SEEDS.index(seed)}')


def train_one(arm,seed):
    start=time.monotonic()
    out=run_name(arm,seed)
    out.mkdir()
    torch.manual_seed(seed)
    policy=Policy() if arm=='a' else r1.BatchPolicy(expose_age=False)
    assert policy.constrained
    captured=[]

    def capture(step):
        assert step not in captured
        policy.model.save_pretrained(out/f'{step:04}',save_embedding_layers=False)
        captured.append(step)

    patients,dev=(load_split(name) for name in ('train','dev'))
    if arm=='a':
        chosen,end,info=r0.optimize(policy,patients,dev,.20,out,seed=seed,rank=rank,capture=capture)
        endpoint=r0.STEPS
    else:
        chosen,end,info=r1.train(policy,r1.cases_for(patients,'train'),r1.cases_for(dev,'dev'),out,
                                seed=seed,rank=rank,capture=capture)
        endpoint=info['steps']
    lock=min(info['dev'],key=lambda d:(-rank(d)[0],-rank(d)[1],d['step']))
    assert lock['step']==info['best_step'] and endpoint==captured[-1]
    assert captured==[row['step'] for row in info['dev']]
    for step,expected in ((info['best_step'],chosen),(endpoint,end)):
        loaded=load_peft_weights(out/f'{step:04}',device='cpu')
        assert set(loaded)==set(expected) and all(torch.equal(loaded[k],v) for k,v in expected.items())
    save(out/'selected.json', dict(arm=arm,seed=seed,step=info['best_step'],end=endpoint,checkpoints=captured,
         training=info,seconds=round(time.monotonic()-start,2),
         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print('training locked',arm,seed,captured[-1],info['best_step'],flush=True)


def read(p):
    return json.loads(Path(p).read_text())


def evaluate_only():
    checkpoints={(arm,seed):read(run_name(arm,seed)/'selected.json') for arm in ARMS for seed in SEEDS}
    assert all(c['arm']==a and c['seed']==s for (a,s),c in checkpoints.items())
    for (arm,seed),row in checkpoints.items():
        saved={p.name for p in run_name(arm,seed).iterdir() if p.is_dir()}
        assert saved=={f'{s:04}' for s in row['checkpoints']}
        assert rank(min(row['training']['dev'],key=lambda d:(-rank(d)[0],-rank(d)[1],d['step'])))==rank(
                next(d for d in row['training']['dev'] if d['step']==row['step']))
    assert not Path('opened').exists()
    Path('opened').touch()
    patients=read('new_private.json')
    assert len(patients)==64
    forms=(*FORMS['test'],*read('forms.json'))
    cases=[dict(view(p,state,q),state=state) for p in patients for state in STATES for q in forms]
    torch.manual_seed(SEEDS[0])
    policy=r1.BatchPolicy(expose_age=False)
    inputs=r1.rows_for(policy,cases)

    def score(mode='trained'):
        value,_=r1.evaluate(policy,cases,inputs,len(forms),mode)
        assert [value['cells'][name]['n'] for name in r1.STRATA]==[384,192,192]
        return value

    controls={name:score(name) for name in ('frozen','rule')}
    assert controls['rule']['counts']['guarded_calls']==1152
    result={arm:{} for arm in ARMS}
    for (arm,seed),lock in checkpoints.items():
        estimates={}
        for mode,step in (('selected',lock['step']),('endpoint',lock['end'])):
            if mode=='endpoint' and step==lock['step']:
                estimates[mode]=estimates['selected']
            else:
                policy.restore(load_peft_weights(run_name(arm,seed)/f'{step:04}',device='cpu'))
                estimates[mode]=score()
        result[arm][str(seed)]=dict(training=lock,test=estimates)
        print('evaluated',arm,seed,flush=True)
    summary={}
    for arm,runs in result.items():
        summary[arm]={}
        for stage in ('selected','endpoint'):
            values=[{**r['test'][stage]['cells'],'overall':r['test'][stage]['counts']} for r in runs.values()]
            summary[arm][stage]={name:{k:dict(mean=mean(v[name][k] for v in values),
                                                   sample_sd=stdev(v[name][k] for v in values))
                                        for k in counts} for name,counts in values[0].items()}
    save(Path('RESULT.json'),dict(seeds=SEEDS,panel=read('panel_receipt.json'), controls=controls,
                                  result=result, statistics=summary,
                                  total_elapsed_seconds=int(time.time())-int(Path('started').read_text())))


if __name__=='__main__':
    if os.environ['PHASE']=='fit':
        train_one(os.environ['ARM'],int(os.environ['TRIAL_SEED']))
    else:
        assert os.environ['PHASE']=='score'
        evaluate_only()
