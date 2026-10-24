"""Pure gate and reporting logic shared by worker and CPU checks."""
import math


def gate(reference,candidate,gold,anchor_reference=None,anchor_candidate=None):
    def valid(rows):return all(x['status']=='completed' and x.get('predicted') in list('ABCDEF') for x in rows)
    def score(rows):return sum(r.get('predicted')==gold[r['id']] for r in rows)
    assert {x['id'] for x in reference}=={x['id'] for x in candidate}
    old=score(reference);new=score(candidate);reasons=[]
    if not valid(reference) or not valid(candidate):reasons.append('incomplete_or_invalid_selection')
    if new<old+1:reasons.append('no_positive_selection_gain')
    base_eligible=not reasons;losses=0;protected=0;anchor_complete=True
    if anchor_reference is not None:
        assert {x['id'] for x in anchor_reference}=={x['id'] for x in anchor_candidate}
        after={x['id']:x for x in anchor_candidate}
        protected=sum(r.get('predicted')==gold[r['id']] for r in anchor_reference)
        losses=sum(r.get('predicted')==gold[r['id']] and after[r['id']].get('predicted')!=gold[r['id']] for r in anchor_reference)
        anchor_complete=valid(anchor_reference) and valid(anchor_candidate)
        if not anchor_complete:reasons.append('incomplete_or_invalid_anchor')
        if losses:reasons.append('old_correct_anchor_loss')
    return {'reference_correct':old,'candidate_correct':new,'denominator':len(reference),'local_infinite_accept':base_eligible,'local_zero_accept':not reasons,'anchor_protected':protected,'anchor_losses':losses,'anchor_complete':anchor_complete,'reasons':reasons}


def paired(before,after,keys,gold):
    a={x['id']:x.get('predicted') for x in before};b={x['id']:x.get('predicted') for x in after}
    cells={'both_correct':0,'gain':0,'loss':0,'both_wrong':0}
    for k in keys:
        x=a.get(k)==gold[k];y=b.get(k)==gold[k]
        cells['both_correct' if x and y else 'gain' if y else 'loss' if x else 'both_wrong']+=1
    n=cells['gain']+cells['loss'];m=min(cells['gain'],cells['loss'])
    cells['mcnemar_exact_two_sided']=min(1.0,2*sum(math.comb(n,k) for k in range(m+1))/2**n) if n else 1.0
    cells['denominator']=len(keys);return cells
