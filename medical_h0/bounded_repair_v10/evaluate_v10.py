"""Evaluator-only; never supplied to the solver or its feedback."""
import bootstrap_v10
from evaluate_v7 import evaluate as previous_evaluate


def evaluate(record,case):
    answer=record.get('answer') or {}
    if case['expected_outcome']=='answerable':
        score=previous_evaluate(record,case)
        score['answerable']=True
    else:
        score={'answerable':False,'numeric_correct':False,'trusted_correct':False,'preliminary_correct':answer.get('status')=='abstain',
               'accepted_computations':len(record['result_ledger']),'state_report':answer.get('status','unfinished'),
               'result_audits':{x['result_id']:{'status':'incorrect','reasons':['required_evidence_'+case['expected_outcome']]} for x in record['result_ledger']}}
    return {**score,'expected_outcome':case['expected_outcome'],'manual_review':'pending'}
