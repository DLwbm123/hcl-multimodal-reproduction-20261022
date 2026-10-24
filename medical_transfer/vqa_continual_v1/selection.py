"""Seeded bipartite allocation: a content group can fill only one role/type slot."""
import random

def allocate(pools,quotas,seed):
    rng=random.Random(seed);adj={}
    for key in sorted(quotas):
        groups=sorted(pools[key]);rng.shuffle(groups);adj[key]=groups
    slots=[(key,i) for key in sorted(quotas,key=lambda k:(len(adj[k]),k)) for i in range(quotas[key])]
    owner={};chosen={}
    def place(slot,seen):
        key,_=slot
        for group in adj[key]:
            if group in seen:continue
            seen.add(group)
            if group not in owner or place(owner[group],seen):
                owner[group]=slot;chosen[slot]=group;return True
        return False
    for slot in slots:
        if not place(slot,set()):raise ValueError('insufficient disjoint groups for frozen quotas: '+str(slot))
    result={k:[] for k in quotas}
    for key,i in slots:
        group=chosen[(key,i)];records=sorted(pools[key][group],key=lambda x:x['id'])
        result[key].append(rng.choice(records))
    assert len(owner)==sum(quotas.values())
    return result
