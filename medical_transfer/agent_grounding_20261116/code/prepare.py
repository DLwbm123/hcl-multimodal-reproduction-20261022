"""Transfer exact never-evaluated partitions, preserving the earlier random split."""
import os,json
from pathlib import Path
R=Path(os.environ['RUN_ROOT']);P=json.loads((R/'protocol.json').read_text());source=Path(P['previous_arbitration_run']);assert not (R/'private_membership.json').exists()
assert json.loads((source/'SUPERVISOR_EXIT.json').read_text())['success'];assert json.loads((source/'SOURCE_GATE.json').read_text())['reasons']==['no_supported_contract_overrides'];assert not any(source.glob('eval_*')) and not any(source.glob('confirm_*'))
learning=json.loads((source/'private_learning.json').read_text());confirmation=json.loads((source/'private_confirmation.json').read_text());member=json.loads((source/'private_membership.json').read_text());assert len(learning['train'])==48 and len(learning['selection'])==24 and len(confirmation)==76
for role,examples in [('selection',learning['selection']),('confirmation',confirmation)]:assert {e['task_id'] for e in examples}=={e['id'] for e in member['members'][role]}
flat=[x for v in member['members'].values() for x in v];assert len(flat)==len({x['group_id'] for x in flat})==len({x['pixel_id'] for x in flat})==100
for name in ['private_learning.json','private_confirmation.json','private_membership.json']:(R/name).write_text((source/name).read_text())
receipt={'reused_train':48,'unchanged_unopened_selection':24,'unchanged_unopened_confirmation':76,'prior_empty_policy':True,'no_prior_eval_or_confirm_workers':True,'partition_seed_preserved':member['seed'],'no_outcome_reallocation':True,'source_pixel_groups_distinct':True,'model_calls':0,'official_test_accessed':False,'patient_independence':'unverified'};(R/'DATA_CHECK.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
