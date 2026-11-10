"""No model imports: verify sealed/unknown rewards and interrupted-pair handling."""
import ast
from pathlib import Path

source=Path(__file__).with_name('worker.py').read_text()
fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='score')
ns={'ARMS':['Fixed','Probe']}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'score','exec'),ns)
inputs=[{'id':n,'question':'synthetic','images':['x'],'composite_image_files':[],
         'semantic_feedback_allowed':n!='unknown','reference_tier':'synthetic'} for n in ['gain','loss','unknown']]
def ep(n,a,p):return {'id':n,'arm':a,'status':'completed','predicted':p,'tool_calls':1,'successful_tools':1,'seconds':2}
episodes=[ep('gain','Fixed','A'),ep('gain','Probe','B'),ep('loss','Fixed','A'),ep('loss','Probe','B'),ep('unknown','Fixed','A'),ep('unknown','Probe','B')]
rows,r=ns['score'](inputs,episodes,{'gain':'B','loss':'A'})
assert r['coverage_complete'] and r['paired']=={'denominator':2,'observed_pairs':2,'gain':1,'loss':1}
assert all(x['benchmark_agreement'] is None for x in rows[-1]['arms'].values())
rows,r=ns['score'](inputs,[e for e in episodes if not(e['id']=='gain' and e['arm']=='Fixed')],{'gain':'B','loss':'A'})
assert not r['coverage_complete'] and r['paired']['gain']==0 and r['paired']['observed_pairs']==1
try:ns['score'](inputs,episodes,{'gain':'B','loss':'A','unknown':'A'})
except AssertionError:pass
else:raise AssertionError('Unknown label was accepted')
try:ns['score'](inputs,episodes+[episodes[0]],{'gain':'B','loss':'A'})
except AssertionError:pass
else:raise AssertionError('Duplicate episode was accepted')
print('Scoring: unknown rewards, missing pairs, duplicate records and full denominators passed')
