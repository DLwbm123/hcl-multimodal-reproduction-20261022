"""Gold is read once both split predictions have been sealed."""
import json
from math import comb


def summarize(root, data, split):
    gold = {r['case']:r['answer'] for r in map(json.loads,(data/split/'answers.jsonl').read_text().splitlines())}
    rows = list(map(json.loads,(root/('private_'+split+'.jsonl')).read_text().splitlines()))
    if {r['case'] for r in rows} != set(gold) or len(rows) != len(gold):
        raise ValueError("incomplete pairing")
    gains = losses = both = different_sets = different_answers = 0
    usage = {a:dict(tool_calls=0, model_forwards=0) for a in ('fixed','adaptive')}
    for row in rows:
        f, a = row['fixed'], row['adaptive']
        target = gold[row['case']]
        fc, ac = f['answer'] == target, a['answer'] == target
        gains += ac and not fc
        losses += fc and not ac
        both += ac and fc
        different_answers += f['answer'] != a['answer']
        different_sets += {r['returned'] for r in f['receipts']} != {r['returned'] for r in a['receipts']}
        for name in usage:
            arm = row[name]
            usage[name]['tool_calls'] += len(arm['receipts'])
            usage[name]['model_forwards'] += arm['selector_forwards'] + arm['diagnosis_forwards']
    discordant = gains + losses
    p = min(1., 2*sum(comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant)
    return dict(n=len(rows), fixed_correct=both+losses, adaptive_correct=both+gains, gained=gains,
                lost=losses, net=gains-losses, both_correct=both, both_wrong=len(rows)-both-discordant,
                different_sets=different_sets, different_answers=different_answers, mcnemar_exact_p=p, usage=usage)
