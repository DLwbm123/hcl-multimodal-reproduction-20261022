"""Greedy execution with shared frozen hints and resumable, exclusive shards."""
import fcntl
import json
import os
from pathlib import Path
import compare as v


def load(name):
    return json.loads(Path(name+'.json').read_text())


if __name__=='__main__':
    assert load('controls')['checked']==1000
    cases=load('cases')
    bank=load('library')
    arms=[('plain',None),('demo',load('demo_gate')['enabled']),('card',load('gate')['enabled'])]
    v.setup_model()
    shards=int(os.environ['SHARDS'])
    for i in range(shards):
        done=Path(f'done_{i}')
        with open(f'lock_{i}','a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            if done.exists(): continue
            for arm,enabled in arms:
                subset=[c for c in cases[i::shards] if enabled is None or v.match(c,bank)['key'] in enabled]
                hint=(lambda c:v.match(c,bank)[arm]) if enabled is not None else None
                v.p.run(subset,f'{arm}_{i}',hint,batch_size=2,hint_budget=1024)
            done.touch()
