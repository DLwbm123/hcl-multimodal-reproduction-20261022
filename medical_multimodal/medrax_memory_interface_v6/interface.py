"""Two independently controlled memory-interface factors; source state stays immutable."""
import copy
import re
from hcl.memory import _memory_candidate_text

ARMS=['H0_static','global_letter','global_semantic','balanced_letter','balanced_semantic']


def option_texts(question):
    matches=list(re.finditer(r'(?m)^\s*([A-F])[.)]\s+',question));result={}
    for i,m in enumerate(matches):
        label=m.group(1)
        if label in result:raise ValueError('Duplicate option label in historical training question')
        end=matches[i+1].start() if i+1<len(matches) else len(question)
        result[label]=question[m.end():end].strip()
    if len(result)<2 or any(not x for x in result.values()):raise ValueError('Historical option text cannot be parsed exactly')
    return result


def select(raw,abstract,question,arm):
    if arm not in ARMS:raise ValueError('Unknown frozen arm')
    if arm=='H0_static':return []
    tokens=lambda x:set(re.findall(r'[a-z0-9]+',x.lower()));query=tokens(question)
    def rank(item):
        words=tokens(_memory_candidate_text(item));u=query|words
        return (-(len(query&words)/len(u) if u else 0),str(item['memory_id']))
    if arm.startswith('balanced_'):
        if not raw or not abstract:raise ValueError('Balanced interface requires both memory kinds')
        return sorted(raw,key=rank)[:1]+sorted(abstract,key=rank)[:1]
    return sorted(raw+abstract,key=rank)[:2]


def render_views(chosen,source_records,arm):
    result=copy.deepcopy(chosen)
    if arm.endswith('_semantic'):
        source={r['memory_id']:r for r in source_records}
        for item in result:
            if item['memory_kind']!='raw':continue
            record=source[item['memory_id']];mapping=option_texts(record['question'])
            for view_field,source_field in [('previous_attempt','response'),('correct_feedback','feedback')]:
                label=str(record[source_field]).strip()
                if label not in mapping:raise ValueError('Historical answer is not an exact option label')
                item[view_field]=mapping[label]
    return result
