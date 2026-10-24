"""Meaningful CPU contracts: selection gain, unoffset anchor losses and retention denominator."""
from logic import gate,paired
from local_update import local_bank
import copy

make=lambda k,a:{'id':k,'predicted':a,'status':'completed'}
gold={'s1':'A','s2':'B','a1':'A','a2':'B'}
old=[make('s1','A'),make('s2','A')];new=[make('s1','A'),make('s2','B')]
anchor=[make('a1','A'),make('a2','A')];changed=[make('a1','B'),make('a2','B')]
v=gate(old,new,gold,anchor,changed)
assert v['local_infinite_accept'] and not v['local_zero_accept'] and v['anchor_losses']==1 and v['anchor_protected']==1
assert not gate(old,old,gold)['local_infinite_accept']
invalid=copy.deepcopy(new);invalid[0]['status']='failed';assert not gate(old,invalid,gold)['local_infinite_accept']
cells=paired(anchor,changed,['a1','a2'],gold);assert cells['gain']==cells['loss']==1 and cells['denominator']==2
empty={'raw':[],'abstract':[],'anchor':[],'skills':[]};proposal=copy.deepcopy(empty)
proposal['raw']=[{'memory_id':str(i),'internal_source':{'task_id':str(i)},'question':str(i)} for i in range(8)]
proposal['abstract']=[{'abstract_role':'main_policy','guidance':'policy'},{'abstract_role':'failure_avoidance','guidance':'avoid'}]
examples=[{'task_id':str(i)} for i in range(8)];state=local_bank(empty,proposal,examples,0)
assert len(state['raw'])==2 and state['raw']==proposal['raw'][:2] and state['abstract']==proposal['abstract'][:1] and empty['raw']==[]
nextproposal=copy.deepcopy(proposal);nextproposal['raw']=[{'memory_id':'b'+str(i),'internal_source':{'task_id':'b'+str(i)},'question':'other '+str(i)} for i in range(8)]
state2=local_bank(state,nextproposal,[{'task_id':'b'+str(i)} for i in range(8)],1)
assert state2['raw'][:2]==state['raw'] and len(state2['raw'])==4 and len(state2['abstract'])==2
print('PASS: positive-gain gate, loss cannot be offset by recovery, invalid cases fail closed, full paired denominator, two-slot local writes and old-state preservation')
