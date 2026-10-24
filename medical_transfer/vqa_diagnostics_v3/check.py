"""No-model preflight of data boundaries, full-bank edits and selection semantics."""
import os,sys,json,copy
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);sys.path[:0]=[str(R/'code'),str(R/'source'),str(R/'source/medical_transfer/src'),str(R/'source/medical_transfer/development_diagnostic'),str(R/'source/medical_transfer/semantic_boundary')]
from diagnostics import flagged,choose_c,choose_gate,shifted_images,contract_text,install_train_retrieval,install_contract,memory_change
from local_update import candidates
from adapter import install_stream
from medical_hcl.bridge import construct_pipeline,NoInferenceModel
from runtime_support import install_protocol
from semantic_boundary import install_semantic_boundary
from hcl.task_interface import TaskInterface
P=json.loads((R/'protocol.json').read_text());data=json.loads((R/'private_data.json').read_text());checks=[]
all_e=[e for x in data.values() for es in x.values() for e in es]
assert len(all_e)==360 and len({x['metadata']['group_id'] for x in all_e})==360 and len({x['metadata']['pixel_id'] for x in all_e})==360
assert all(x['metadata']['role'] not in ['test','reserved_report'] for x in all_e)
checks.append('360_unique_groups_and_pixels_no_prior_report_or_test')
assert flagged({'question':'none','answer':'nan'}) and not flagged({'question':'Where is it?','answer':'none'})
assert flagged({'question':'Where is it?','answer':'nan'})
checks.append('missing_fields_flagged_absence_none_not_excluded')
original=data['B']['selection'];shifted=shifted_images(original)
assert all(a['image']!=b['image'] and a['task_id']==b['task_id'] and a['answer']==b['answer'] for a,b in zip(original,shifted))
# Even with a cached structured task, runtime envelope is rebuilt from current files.
task=TaskInterface();e=shifted[0];chunk=task._build_runtime_envelope(e,0,hcl_interface=task._rule_hcl_interface(e),interface_mode='cpu')
assert any(str(e['image'])==str(x.get('path')) for x in chunk['task_context']['files'])
checks.append('cyclic_images_keep_questions_targets_and_runtime_uses_shifted_file')
backend=NoInferenceModel();backend.scope='cpu';pipe=construct_pipeline(os.environ['PREPARED_DATA'],R/'cpu_check',model=backend,real_execution_authorized=True)
install_protocol(pipe,backend,R/'cpu_check',P);install_semantic_boundary(pipe);install_stream(pipe,'stability',P)
common=json.loads((R/'private_after_A.json').read_text());pipe.memory.restore(common)
assert len(common['raw'])==10 and len(common['abstract'])==2
prop=copy.deepcopy(common)
for i in range(3):prop['raw'][i]={**prop['raw'][i],'memory_id':'new'+str(i),'internal_source':{'task_id':'new'+str(i)},'question':common['raw'][i]['question']}
prop['abstract'][0]={**prop['abstract'][0],'guidance':'synthetic updated'}
train=[{'task_id':'new'+str(i)} for i in range(3)]
local=dict(candidates(common,prop,train,0,True))['raw_and_abstract'];changes=memory_change(common,local)
assert len(local['raw'])==10 and changes['raw']['new_ids']==2 and changes['raw']['removed_ids']==2 and changes['raw']['unchanged']==8
assert changes['abstract']['modified_existing']==1 and local['anchor']==common['anchor']
checks.append('full_bank_local_replaces_two_preserves_eight_and_one_abstract')
ref={'current_task_metrics':{'primary_score':0,'correct':0,'format_compliance_rate':1.0}}
safe={'current_task_metrics':{'primary_score':1/24,'correct':1,'format_compliance_rate':1.0},'historical_anchor_metrics':{'forget_count':0}}
unsafe={'current_task_metrics':{'primary_score':2/24,'correct':2,'format_compliance_rate':1.0},'historical_anchor_metrics':{'forget_count':1}}
assert choose_gate(pipe,ref,[unsafe,safe],0)[0]==1
assert choose_gate(pipe,ref,[unsafe,safe],None)[0]==0
assert pipe.optimizer.historical_loss_budget==0
checks.append('same_predictions_distinguish_hard_budgets_and_restore_setting')
base={'quality':{'correct':0,'flag_free_correct':0},'metrics':{'format_compliance_rate':1.0}}
results={name:copy.deepcopy(base) for name in ['after_B','local_raw','bulk_raw','bulk_raw_train_only_lexical','bulk_raw_train_only_lexical_contract']}
results['local_raw']['quality']['correct']=3
assert choose_c(results) is None
results['bulk_raw']['quality'].update(correct=2,flag_free_correct=2)
results['bulk_raw_train_only_lexical']['quality'].update(correct=2,flag_free_correct=2)
assert choose_c(results)=='bulk_raw'
checks.append('C_selection_requires_gain_outside_missing_fields_and_fixed_tie_order')
allowed={common['raw'][0]['memory_id']};install_train_retrieval(pipe,allowed)
items,trace=pipe.memory.select_for_context(current_input={'question':common['raw'][0]['question']},workflow_decision={})
assert len(items)==1 and items[0]['memory_id'] in allowed
text=contract_text(data['C']['train'][:24]);assert 'center' in text and 'nan' not in text
install_contract(pipe,text)
checks.append('forced_retrieval_train_allowlist_and_training_only_contract')
assert backend.calls==0
result={'passed':True,'checks':checks,'model_calls':0};(R/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
