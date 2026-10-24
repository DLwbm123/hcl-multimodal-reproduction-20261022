"""No new inference: exact option mapping and orthogonal retrieval/representation contracts."""
import copy
from interface import select,render_views,option_texts
assert option_texts('Question?\nA. First line\n continuation\nB. Second')=={'A':'First line\n continuation','B':'Second'}
source=[{'memory_id':'r1','question':'Question?\nA) First\nB) Second','response':'B','feedback':'B'}]
raw=[{'memory_id':'r1','memory_kind':'raw','question':'Question?','previous_attempt':'B','correct_feedback':'B'},{'memory_id':'r2','memory_kind':'raw','question':'Question more','previous_attempt':'A','correct_feedback':'A'}]
source.append({'memory_id':'r2','question':'More?\nA) Earlier\nB) Later','response':'A','feedback':'A'})
abstract=[{'memory_id':'a1','memory_kind':'abstract','guidance':'Inspect evidence'}];before=copy.deepcopy((raw,abstract,source))
for retrieval in ['global','balanced']:
    left=select(raw,abstract,'Question?',retrieval+'_letter');right=select(raw,abstract,'Question?',retrieval+'_semantic');assert left==right
    text=render_views(right,source,retrieval+'_semantic');assert [x['memory_id'] for x in text]==[x['memory_id'] for x in right]
    assert all(x.get('correct_feedback') not in ['A','B'] for x in text if x['memory_kind']=='raw')
    if retrieval=='balanced':assert [x['memory_kind'] for x in text]==['raw','abstract']
assert (raw,abstract,source)==before and select(raw,abstract,'Q','H0_static')==[]
try:option_texts('No valid labelled alternatives')
except ValueError:pass
else:raise AssertionError('Invalid historical option mapping must fail closed')
print('PASS: exact multiline option mapping, same chosen IDs across representation arms, abstract quota, immutable source, unchanged empty baseline, invalid mapping fail closed')
