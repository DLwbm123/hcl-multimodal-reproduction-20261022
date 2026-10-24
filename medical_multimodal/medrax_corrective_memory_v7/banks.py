"""A single predeclared write-source contrast; report labels never enter here."""
import copy

ARMS=['H0_static','success_memory','corrective_memory']
EMPTY={'raw':[],'abstract':[],'anchor':[],'skills':[]}


def matched_banks(records,inputs,predictions,gold):
    by_pred={x['id']:x for x in predictions};by_record={x['internal_source']['task_id']:x for x in records}
    states={name:copy.deepcopy(EMPTY) for name in ARMS};audit={}
    for stage in ['A','B']:
        rows=[x for x in inputs if x['role']=='train' and x['stage']==stage];good=[];bad=[]
        for row in rows:
            p=by_pred[row['id']]
            if p['status']!='completed' or p.get('predicted') not in list('ABCDEF'):raise ValueError('Incomplete training; no fabricated correction')
            record=by_record[row['id']]
            assert record['feedback']==gold[row['id']] and record['response']==p['predicted']
            (good if p['predicted']==gold[row['id']] else bad).append(record)
        k=min(2,len(good),len(bad))
        states['success_memory']['raw'].extend(copy.deepcopy(good[:k]));states['corrective_memory']['raw'].extend(copy.deepcopy(bad[:k]))
        audit[stage]={'training_denominator':len(rows),'correct':len(good),'incorrect':len(bad),'matched_selected_per_bank':k}
    assert len(states['success_memory']['raw'])==len(states['corrective_memory']['raw'])<=4
    return states,audit
