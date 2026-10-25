"""One bounded H0 repeat panel and fixed training pool; no method search."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import traceback
import run as core
from logic import paired
from qualification import first_errors, source_summary, valid
from storage import tree_bytes

ROOT = core.ROOT; BUDGET = core.BUDGET
os.environ['TMPDIR'] = '/data/bmw/workers/t1004'
os.environ['TRITON_CACHE_DIR'] = '/data/bmw/workers/t1004/triton'
INPUTS = []; GOLD = {}; SELECTED = []; FATAL = None


def watchdog():
    roots = [ROOT] + [ROOT.parent / n for n in ['medrax_subscription_v4', 'medrax_hcl_v5', 'medrax_memory_interface_v6', 'medrax_corrective_memory_v7', 'medrax_error_abstraction_v8']]
    while True:
        try:
            sizes = [tree_bytes(p) for p in roots]
            core.save('RUN_GUARD.json', {'time': time.time(), 'own_bytes': sizes[0], 'aggregate_bytes': sum(sizes), 'deadline': BUDGET['hard_deadline_epoch'], 'probe': 'python_lstat_no_inner_symlinks'})
            if sizes[0] >= BUDGET['own_stop_new_bytes'] or sum(sizes) >= BUDGET['aggregate_stop_new_bytes'] or time.time() >= BUDGET['hard_deadline_epoch']:
                raise RuntimeError('Original time or storage stop')
        except BaseException as exc:
            try: core.save('STOPPED.json', {'time': time.time(), 'error': repr(exc), 'context': core.CONTEXT})
            finally: os.killpg(os.getpgrp(), signal.SIGTERM)
            return
        time.sleep(15)


def finalize():
    rows = lambda label: [x for x in core.EPISODES if x['state'] == label]
    repeats = [x for x in INPUTS if x['role'] == 'repeat_development']
    train = [x for x in INPUTS if x['role'] == 'train']
    first = rows('H0_repeat_1'); second = rows('H0_repeat_2')
    by1 = {x['id']: x for x in first}; by2 = {x['id']: x for x in second}
    def complete_rows(by):
        return [by.get(x['id'], {'id': x['id'], 'predicted': None, 'status': 'not_run'}) for x in repeats]
    reliability = {}
    for stage in ['A', 'B']:
        keys = [x['id'] for x in repeats if x['stage'] == stage]
        both_valid = [k for k in keys if valid(by1.get(k, {})) and valid(by2.get(k, {}))]
        reliability[stage] = {'denominator': len(keys), 'both_valid': len(both_valid),
                              'answer_disagreement_both_valid': sum(by1[k]['predicted'] != by2[k]['predicted'] for k in both_valid),
                              'missing_or_invalid_pairs': len(keys) - len(both_valid),
                              'correctness_paired_full_denominator': paired(complete_rows(by1), complete_rows(by2), keys, GOLD)}
    sources = {s: source_summary([x for x in train if x['stage'] == s], rows('H0_train'), rows('H0_error_recheck'), [x for x in SELECTED if x['stage'] == s], GOLD) for s in ['A', 'B']}
    flips = {}
    for label in ['H0_repeat_1', 'H0_repeat_2', 'H0_train', 'H0_error_recheck']:
        parsed = []
        for row in rows(label):
            match = re.match(r'([A-F])(?:[).:\s]|$)', row.get('initial_final', '').lstrip().lstrip('*'))
            if match and valid(row): parsed.append(match.group(1) != row['predicted'])
        flips[label] = {'episodes': len(rows(label)), 'valid': sum(valid(r) for r in rows(label)), 'initial_parseable_valid': len(parsed), 'initial_final_changed': sum(parsed)}
    expected = 96 + len(SELECTED)
    result = {'time': time.time(), 'fatal': FATAL, 'mandatory_episodes': 96, 'max_episodes': 104,
              'planned_given_selected_errors': expected, 'physical_episodes': len(core.EPISODES),
              'complete': FATAL is None and len(core.EPISODES) == expected and all(valid(x) for x in core.EPISODES),
              'repeatability': reliability, 'sources': sources, 'initial_final_consistency': flips,
              'confirmation_reserved_cases': 96, 'confirmation_labels_accessed': False,
              'medical_source_validity_established': False, 'continual_learning_established': False,
              'seconds_since_original_start': time.time() - BUDGET['started_epoch']}
    core.save('AGGREGATE_REPORT.json', result); core.save('FINISHED.json', result)


def main():
    global INPUTS, GOLD, SELECTED, FATAL
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Independent watchdog termination')))
    threading.Thread(target=watchdog, daemon=True).start()
    core.save('STARTED.json', {'time': time.time(), 'original_start': BUDGET['started_epoch'], 'deadline': BUDGET['hard_deadline_epoch']})
    INPUTS = json.loads((ROOT / 'private/inputs.json').read_text())
    GOLD = json.loads((ROOT / 'private/supervision.json').read_text())
    from hcl.memory import ExperienceMemory, MemoryConfig
    memory = ExperienceMemory(MemoryConfig(record_dir=str(ROOT / 'private/memory'), raw_capacity_per_task=4, abstract_capacity_per_task=2, skill_enabled=False, anchor_selection_mode='off'))
    try:
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        available = {int(x.split(',')[0]): int(x.split(',')[1]) for x in free.strip().splitlines()}
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
        core.save('GPU_PREFLIGHT.json', {'time': time.time(), 'gpu': gpu, 'free_mib': available[gpu]})
        if available[gpu] < 45000: raise RuntimeError('Insufficient headroom; no unrelated process touched')
        core.CONTEXT['phase'] = 'loading_six_tools'
        tools, state, Agent, prompt = core.rt.load_six()
        core.save('TOOLS_LOADED.json', {'time': time.time(), 'names': [x.name for x in tools]})
        def run(items, label):
            return [core.episode(x, core.EMPTY, label, tools, state, Agent, prompt, memory, set()) for x in items]
        repeats = [x for x in INPUTS if x['role'] == 'repeat_development']
        train = [x for x in INPUTS if x['role'] == 'train']
        run(repeats, 'H0_repeat_1'); run(list(reversed(repeats)), 'H0_repeat_2')
        initial = run(train, 'H0_train')
        for stage in ['A', 'B']:
            SELECTED.extend(first_errors([x for x in train if x['stage'] == stage], initial, GOLD, 4))
        core.save('private/error_recheck_members.json', [{k: x[k] for k in ['id', 'case_id', 'stage']} for x in SELECTED])
        core.save('RECHECKS_FROZEN.json', {'time': time.time(), 'count': len(SELECTED), 'rule': 'first_four_valid_errors_per_stage_in_frozen_train_order'})
        run(SELECTED, 'H0_error_recheck')
        assert memory.snapshot() == core.EMPTY and all(x['context_chars'] == 0 and not x['retrieved_ids'] for x in core.EPISODES)
    except BaseException as exc:
        FATAL = repr(exc); core.save('FAILED.json', {'time': time.time(), 'error': FATAL, 'traceback': traceback.format_exc(), 'context': core.CONTEXT})
    finally: finalize()
    print(json.dumps({'finished': True, 'fatal': FATAL, 'episodes': len(core.EPISODES)}), flush=True)


if __name__ == '__main__': main()
