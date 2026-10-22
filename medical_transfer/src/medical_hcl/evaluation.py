"""Strict Kvasir metrics; all expected cases remain in the denominator."""
import collections
import json
import math
import re
import unicodedata

TASKS = ('classification', 'polyp_localization', 'visual_qa')
SCHEMA = 'medical_native_fixed_denominator_v1'

def normalize(x):
    return ' '.join(unicodedata.normalize('NFKC', x).casefold().split())

def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def parse_bbox(value, width, height):
    if isinstance(value, str):
        try: value = json.loads(value)
        except (ValueError, TypeError): return None
    if not isinstance(value, dict) or set(value) != {'bbox'}: return None
    box = value['bbox']
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(number(x) for x in box): return None
    x1,y1,x2,y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height): return None
    return tuple(float(x) for x in box)

def iou(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.0

def qa_kind(answer, question_type):
    if ';' in answer: return 'set'
    if question_type == 'count' and re.fullmatch(r'\d+', normalize(answer)): return 'integer'
    return 'text'

def qa_value(answer, kind):
    if not isinstance(answer,str) or not answer.strip() or len(answer) > 4096: return None
    value=normalize(answer)
    if kind == 'integer': return int(value) if re.fullmatch(r'\d+',value) else None
    if kind == 'set':
        parts=[normalize(x) for x in answer.split(';')]
        return frozenset(parts) if all(parts) else None
    return value

def token_f1(a,b):
    ca,cb=collections.Counter(normalize(a).split()),collections.Counter(normalize(b).split())
    shared=sum((ca&cb).values())
    return 2*shared/(sum(ca.values())+sum(cb.values())) if ca or cb else 1.0

def score_case(example, prediction, vocabulary):
    """Return (valid, anchor_success, task_score, auxiliary_score)."""
    task=example['task_name'];gold=example['answer'];meta=example['metadata']
    if prediction is None:return False,False,0.0,0.0
    if task=='classification':
        valid=isinstance(prediction,str) and prediction.strip() in vocabulary
        correct=valid and prediction.strip()==gold
        return valid,correct,float(correct),0.0
    if task=='polyp_localization':
        box=parse_bbox(prediction,meta['width'],meta['height'])
        if box is None:return False,False,0.0,0.0
        overlap=iou(box,gold['bbox'])
        return True,overlap>=0.5,float(overlap>=0.5),overlap
    if task=='visual_qa':
        kind=meta['answer_kind'];value=qa_value(prediction,kind);expected=qa_value(gold,kind)
        valid=value is not None
        exact=valid and value==expected
        if not valid:return False,False,0.0,0.0
        if kind=='set':
            aux=2*len(value&expected)/(len(value)+len(expected))
        else:aux=token_f1(str(prediction),gold)
        return True,exact,float(exact),aux
    raise ValueError('unknown task')

def validate_gold(example, vocabulary):
    t=example['task_name'];a=example.get('answer');m=example.get('metadata',{})
    if t=='classification':ok=isinstance(a,str) and a in vocabulary
    elif t=='polyp_localization':ok=parse_bbox(a,m['width'],m['height']) is not None
    elif t=='visual_qa':ok=isinstance(a,str) and qa_value(a,m['answer_kind']) is not None
    else:ok=False
    if not ok:raise ValueError('invalid evaluation target: '+str(example.get('task_id')))

def behavior(prediction):
    return (prediction.get('answer'),prediction.get('status','ok'),bool(prediction.get('error')))

def index_predictions(predictions, known, allow_superset=False):
    result={}
    for p in predictions:
        key=p.get('task_id')
        if not isinstance(key,str) or key in result:raise ValueError('duplicate/missing prediction ID')
        if key not in known and not allow_superset:raise ValueError('unknown prediction ID')
        result[key]=p
    return result

class MedicalEvaluator:
    def __init__(self,vocabulary):
        self.vocabulary=tuple(vocabulary)
        if not self.vocabulary or len(set(self.vocabulary))!=len(self.vocabulary):raise ValueError('bad vocabulary')

    def evaluate(self,examples,predictions,*,split='',reference_predictions=None,allow_superset=False):
        ids=[x['task_id'] for x in examples]
        if len(ids)!=len(set(ids)):raise ValueError('duplicate expected ID')
        by=index_predictions(predictions,set(ids),allow_superset)
        ref=index_predictions(reference_predictions or [],set(ids),allow_superset)
        rows=[];conf=collections.Counter();per_task=collections.defaultdict(list)
        for e in examples:
            validate_gold(e,self.vocabulary)
            key=e['task_id'];p=by.get(key,{})
            # An explicit timeout/error wins over any stale answer in that record.
            answer=None if p.get('status','ok')!='ok' or p.get('error') else p.get('answer')
            valid,success,score,aux=score_case(e,answer,self.vocabulary)
            rp=ref.get(key,{})
            reference=None if rp.get('status','ok')!='ok' or rp.get('error') else rp.get('answer')
            rv,rs,rscore,_=score_case(e,reference,self.vocabulary)
            row={'task_id':key,'task':e['task_name'],'valid':bool(valid),'success':bool(success),'score':score,'auxiliary':aux,'reference_valid':rv,'reference_success':rs,'reference_score':rscore,'missing':key not in by}
            rows.append(row);per_task[e['task_name']].append(row)
            if e['task_name']=='classification':conf[(e['answer'],answer.strip() if valid else '<invalid>')]+=1
        n=len(rows);valid=sum(x['valid'] for x in rows);correct=sum(x['success'] for x in rows)
        tasks=set(per_task);single=next(iter(tasks)) if len(tasks)==1 else None
        primary=sum(x['score'] for x in rows)/n if n and single else None
        names={'classification':'accuracy','polyp_localization':'localization_accuracy_iou50','visual_qa':'normalized_exact_match'}
        metrics={'evaluation_schema_version':SCHEMA,'total':n,'correct':correct,'format_compliant_count':valid,'format_compliance_rate':valid/n if n else 0.0,'missing_count':sum(x['missing'] for x in rows),'primary_metric_name':names.get(single,'empty' if not rows else 'mixed_tasks_not_comparable'),'primary_score':primary,'mean_example_score':primary,'has_reference':reference_predictions is not None,'reference_correct':sum(x['reference_success'] for x in rows),'forget_count':sum(x['reference_success'] and not x['success'] for x in rows),'recovered_count':sum(not x['reference_success'] and x['success'] for x in rows) if reference_predictions is not None else 0,'behavior_comparable_count':n if reference_predictions is not None else 0,'behavior_change_count':sum(behavior(by.get(k,{}))!=behavior(ref.get(k,{})) for k in ids) if reference_predictions is not None else 0,'task_primary_scores':{t:sum(x['score'] for x in rs)/len(rs) for t,rs in per_task.items()}}
        metrics['forget_rate']=metrics['forget_count']/metrics['reference_correct'] if metrics['reference_correct'] else 0.0
        if reference_predictions is not None:
            metrics['reference_primary_score']=sum(x['reference_score'] for x in rows)/n if n and single else None
            metrics['primary_score_delta']=primary-metrics['reference_primary_score'] if primary is not None else None
        if single=='classification':
            f1=[];support={}
            for label in self.vocabulary:
                tp=conf[(label,label)];fp=sum(v for (gold,p),v in conf.items() if p==label and gold!=label);fn=sum(v for (gold,p),v in conf.items() if gold==label and p!=label)
                f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
                support[label]=sum(v for (gold,p),v in conf.items() if gold==label)
            metrics.update(macro_f1=sum(f1)/len(f1),macro_f1_classes=len(f1),class_support=support)
        if single=='polyp_localization':metrics['mean_iou']=sum(x['auxiliary'] for x in rows)/n
        if single=='visual_qa':metrics['answer_f1_diagnostic']=sum(x['auxiliary'] for x in rows)/n
        return {'split':split,'metrics':metrics,'predictions':predictions,'case_results':rows}

class HCLMedicalEvaluator(MedicalEvaluator):
    """The upstream gate evaluates subsets using one shared prediction collection."""
    def evaluate(self,*args,**kwargs):
        return super().evaluate(*args,**kwargs,allow_superset=True)

def stage_summary(matrix):
    """Triangular task-native [0,1] R; no best-test checkpoint selection."""
    n=len(matrix)
    if not n or any(len(row)!=n for row in matrix):raise ValueError('square stage matrix required')
    for t,row in enumerate(matrix):
        for j,x in enumerate(row):
            if j<=t and (not number(x) or not 0<=x<=1):raise ValueError('missing/nonfinite seen task score')
            if j>t and x is not None:raise ValueError('unseen task score must be null')
    final=matrix[-1]
    forgetting=[max(matrix[t][j] for t in range(j,n-1))-final[j] for j in range(n-1)]
    return {'final_average_percent':100*sum(final)/n,'average_forgetting_pp':100*sum(forgetting)/(n-1) if n>1 else 0.0,'bwt_pp':100*sum(final[j]-matrix[j][j] for j in range(n-1))/(n-1) if n>1 else 0.0,'task_forgetting_pp':[100*x for x in forgetting],'scope':'descriptive mean of distinct task success rates; original metrics retained'}
