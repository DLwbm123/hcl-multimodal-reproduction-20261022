"""Summarize frozen runs without exporting cases, answers, or tool descriptions."""
import json
import sys
from collections import Counter
from pathlib import Path


def summarize(rows):
    summary={}
    for arm in ('seed','learned'):
        records=[r[arm] for r in rows]
        empty=[r for r in records if not any(e['accepted'] for e in r['events'])]
        events=[e for r in records for e in r['events']]
        summary[arm]=dict(cases_without_accepted_evidence=len(empty),
                          correct_without_accepted_evidence=sum(r['correct'] for r in empty),
                          cases_at_cap=sum(len(r['events'])==4 for r in records),
                          declined_changed_drafts=sum(not e['accepted'] and e['changed_draft'] for e in events),
                          adopted_changed_drafts=sum(e['accepted'] and e['changed_draft'] for e in events))
    summary['same_final_answer_as_seed']=sum(r['seed']['answer']==r['learned']['answer'] for r in rows)
    for direction,cmp in [('gain',1),('loss',-1)]:
        selected=[r['learned'] for r in rows if r['learned']['correct']-r['baseline']['correct']==cmp]
        summary[direction+'_vs_base_final_accepted_tools']=dict(Counter(
            next((e['tool'] for e in reversed(r['events']) if e['accepted']),'(none)') for r in selected))
    return summary


if __name__=='__main__':
    root=Path(sys.argv[1])
    result=json.loads((root/'result.json').read_text())
    totals=Counter()
    for counts in result['fit_by_tool'].values():
        totals.update(counts)
    report={'fit_effect_totals':dict(totals)}
    for phase in ('gate','test'):
        path=root/('private_'+phase+'.json')
        if path.exists():
            report[phase]=summarize(list(json.loads(path.read_text()).values()))
    print(json.dumps(report,indent=2))
