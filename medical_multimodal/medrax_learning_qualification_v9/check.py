"""CPU checks of data isolation, frozen selection and failure denominators."""
import json
import os
from pathlib import Path
from qualification import first_errors, source_summary


def main():
    root = Path(os.environ['R981_ROOT'])
    read = lambda name: json.loads((root / name).read_text())
    inputs = read('private/inputs.json'); members = read('private/members.json'); gold = read('private/supervision.json')
    reserved = [x for x in members if x['role'] == 'confirmation_reserved']
    assert len(inputs) == len({x['id'] for x in inputs}) == len(gold) == 72
    assert len(reserved) == len({x['case_id'] for x in reserved}) == 96
    assert not {str(x['case_id']) for x in reserved} & {x['case_id'] for x in inputs}
    assert not {x['full_question_id'] for x in reserved} & set(gold)
    assert set(gold) == {x['id'] for x in inputs}
    assert all(set(x) == {'id', 'case_id', 'stage', 'role', 'question', 'images'} for x in inputs)
    for stage in ['A', 'B']:
        assert sum(x['stage'] == stage and x['role'] == 'train' for x in inputs) == 24
        assert sum(x['stage'] == stage and x['role'] == 'repeat_development' for x in inputs) == 12
    from hcl.memory import ExperienceMemory, MemoryConfig
    import run as core
    memory = ExperienceMemory(MemoryConfig(record_dir=str(root / 'private/cpu_memory'), skill_enabled=False, anchor_selection_mode='off'))
    for row in inputs:
        memory.restore(core.EMPTY)
        assert core.memory_context(memory, row['question'], set()) == ('', [])
    items = [{'id': str(i)} for i in range(8)]; labels = {x['id']: 'A' for x in items}
    initial = [{'id': str(i), 'status': 'completed', 'predicted': 'B'} for i in range(8)]
    initial[1]['status'] = 'failed'; initial[2]['predicted'] = None; initial[3]['predicted'] = 'A'
    selected = first_errors(items, initial, labels)
    assert [x['id'] for x in selected] == ['0', '4', '5', '6']
    checks = [{'id': '0', 'status': 'completed', 'predicted': 'B'}, {'id': '4', 'status': 'completed', 'predicted': 'C'}, {'id': '5', 'status': 'completed', 'predicted': 'A'}]
    summary = source_summary(items, initial, checks, selected, labels)
    assert summary['training_missing_or_invalid'] == 2 and summary['wrong_twice'] == 2
    assert summary['same_wrong_option'] == summary['different_wrong_option'] == summary['recovered'] == summary['recheck_missing_or_invalid'] == 1
    assert summary['preliminary_one_bank_sources'] and not summary['preliminary_two_disjoint_banks_sources']
    value = {'passed': True, 'input_cases': 72, 'reserved_cases': 96, 'empty_contexts_checked': 72, 'confirmation_gold_absent': True, 'selection_and_failure_denominators_checked': True}
    (root / 'CPU_CHECK.json').write_text(json.dumps(value, indent=2)); print(json.dumps(value))


if __name__ == '__main__': main()
