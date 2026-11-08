"""Continuous aggregate progress and terminal full-denominator report."""
import json
import os
from pathlib import Path
import time
from policy import summarize

ROOT = Path(os.environ['R1136_ROOT'])
cfg = json.loads((ROOT/'deployment.private.json').read_text())
budget = json.loads((ROOT/'budget.json').read_text())
inputs = json.loads((ROOT/'private/inputs.json').read_text())


def save(name, value):
    path = ROOT/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


while True:
    episodes, status = [], []
    for partition in cfg['partitions']:
        root = Path(partition['remote_root'])
        rows = json.loads((root/'private/episodes.json').read_text()) if (root/'private/episodes.json').exists() else []
        episodes.extend(rows)
        terminal = json.loads((root/'EXIT.json').read_text()) if (root/'EXIT.json').exists() else None
        error = json.loads((root/'FAILED.json').read_text()).get('error') if (root/'FAILED.json').exists() else None
        status.append({'partition': partition['name'], 'gpu': partition['gpu'], 'episodes': len(rows),
                       'tools_loaded': (root/'TOOLS_LOADED.json').exists(), 'exit': terminal, 'error': error})
    save('PROGRESS.json', {'time': time.time(), 'recorded_episodes': len(episodes), 'planned_episodes': 96,
                           'partitions': status})
    if all(x['exit'] is not None for x in status) or time.time() >= budget['hard_deadline_epoch']+30:
        gold = json.loads((ROOT/'private/sealed_gold.json').read_text())
        report = summarize(inputs, episodes, gold)
        report.update(finished_at=time.time(), partitions=status,
                      seconds_since_original_start=time.time()-budget['started_epoch'],
                      complete=len(episodes) == 96 and all(x['status'] == 'completed' and x['predicted'] is not None for x in episodes),
                      execution_amendment='User requested GPUs 5 and 6; 3 partitions of 8 sources, same within-case arm order; no prior episodes restarted.')
        save('private/episodes.json', episodes)
        save('AGGREGATE_REPORT.json', report)
        save('FINISHED.json', report)
        break
    time.sleep(5)
