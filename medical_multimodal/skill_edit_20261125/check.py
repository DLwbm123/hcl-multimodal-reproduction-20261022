"""Regression checks for atomic edits, schema repair, prior-state isolation."""
import copy
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get("PILOT_ROOT", root.parent / "skill_pilot_20261116")) / "code"))
from proposal import apply_edits, propose

skill = {"name": "demo_box", "task": "T1", "description": "Training fixture",
         "applies_when": "A box is requested.", "steps": [{"id": "a", "tool": "localize", "args": {}}],
         "output": {"bbox": "$a.bbox"}}
bad = dict(skill, applies_when="")
log = []
responses = iter(json.dumps({"skills": [s]}) for s in (bad, skill))
def generate(prompt):
    if log:
        assert "applies_when must be a nonblank string" in prompt
    return next(responses)
assert propose(generate, "T1", [], log=lambda *row: log.append(row)) == skill
assert len(log) == 2
assert apply_edits(None, {"edits": [{"op": "create", "value": skill}]}, "T1") == skill
before = copy.deepcopy(skill)
insert = {"op": "insert_step", "index": 1, "value": {"id": "b", "tool": "expand_box", "args": {"bbox": "$a.bbox", "fraction": 0.1}}}
redirect = {"op": "replace", "field": "output", "value": {"bbox": "$b.bbox"}}
valid = apply_edits(skill, {"edits": [insert, redirect]}, "T1")
assert valid["output"] == redirect["value"] and len(valid["steps"]) == 2
for edits in ([redirect], [insert, redirect, redirect], [{"op": "delete_step", "index": 0}], [{"op": "create", "value": skill}]):
    try:
        apply_edits(skill, {"edits": edits}, "T1")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid edit accepted")
assert skill == before
print(json.dumps({"passed": True, "scope": "schema repair, cold start, bounded atomic edits and rollback"}))
