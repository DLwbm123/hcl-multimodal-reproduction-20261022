"""Training-only skill proposals for the existing HCL executable skill DSL."""
import copy
import json

from core import CATALOG, SCHEMA, TASKS, decode, validate_library


def check_skill(skill, task):
    if not isinstance(skill, dict) or skill.get("task") != task:
        raise ValueError(f"expected one skill for task {task}")
    for field in ("description", "applies_when"):
        value = skill.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 600:
            raise ValueError(f"{field} must be a nonblank string of 1..600 characters")
    return validate_library([skill])[0]


EDIT_FORMAT = """Return {"edits":[...]} with one or two atomic operations, in order.
With no draft use exactly one {"op":"create","value":<full skill object>}.
With a draft, allowed forms:
{"op":"replace","field":"description|applies_when|output","value":...}
{"op":"insert_step","index":0,"value":<step>}
{"op":"replace_step","index":0,"value":<step>}
{"op":"delete_step","index":0}
Indices are zero-based in the current intermediate list. Choose one literal field per edit.
The initial creation is a bootstrap; subsequent edits cannot replace an entire skill."""


def apply_edits(draft, response, task):
    if set(response) != {"edits"} or not isinstance(response["edits"], list):
        raise ValueError('expected {"edits":[...]}')
    edits = response["edits"]
    if not 1 <= len(edits) <= 2:
        raise ValueError("expected one or two edits")
    skill = copy.deepcopy(draft)
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("each edit must be an object")
        op = edit.get("op")
        if skill is None:
            if op != "create" or len(edits) != 1 or set(edit) != {"op", "value"}:
                raise ValueError("empty draft requires a single create operation")
            skill = edit["value"]
        elif op == "replace":
            if set(edit) != {"op", "field", "value"} or edit["field"] not in ("description", "applies_when", "output"):
                raise ValueError("replace requires one permitted field and value")
            skill[edit["field"]] = edit["value"]
        elif op in ("insert_step", "replace_step", "delete_step"):
            keys = {"op", "index"} if op == "delete_step" else {"op", "index", "value"}
            index = edit.get("index")
            limit = len(skill["steps"]) + int(op == "insert_step")
            if set(edit) != keys or type(index) is not int or not 0 <= index < limit:
                raise ValueError("invalid step index or edit fields")
            if op == "insert_step":
                skill["steps"].insert(index, edit["value"])
            elif op == "delete_step":
                skill["steps"].pop(index)
            else:
                skill["steps"][index] = edit["value"]
        else:
            raise ValueError(f"unsupported operation {op}")
    if skill == draft:
        raise ValueError("edits did not change the draft")
    return check_skill(skill, task)


def propose(generate, task, evidence, draft=None, mode="full", log=None):
    if mode not in ("full", "edit"):
        raise ValueError("unknown proposer")
    schema = SCHEMA[SCHEMA.index("Every skill"):SCHEMA.index("Generate procedures")]
    envelope = EDIT_FORMAT if mode == "edit" else 'Return {"skills":[<one skill object>]}.'
    prompt = (
        "Study the successful and unsuccessful TRAINING tool trajectories below. Identify a "
        "procedural error and design a reusable correction. Prefer fewer calls if equally grounded. "
        "Preserve useful steps. Never copy image coordinates, mask IDs or answers. "
        "Return only the requested JSON. description and applies_when must each be a nonempty "
        "plain-language string (1..600 characters); state concretely when the controller should invoke it. "
        "Each step id is local to this skill.\n" + schema + "\n" + json.dumps(CATALOG)
        + "\nTask: " + task + " " + TASKS[task] + "\nDraft: " + json.dumps(draft)
        + "\nTraining evidence: " + json.dumps(evidence) + "\n" + envelope
    )
    for attempt in range(2):
        output = generate(prompt)
        try:
            response = decode(output)
            if mode == "edit":
                candidate = apply_edits(draft, response, task)
            else:
                if set(response) != {"skills"} or not isinstance(response["skills"], list) or len(response["skills"]) != 1:
                    raise ValueError("expected a skills array with exactly one object")
                candidate = check_skill(response["skills"][0], task)
                if candidate == draft:
                    raise ValueError("unchanged draft")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            error = str(exc)
            if log:
                log(attempt, output, error)
            prompt += "\nRejected: " + error + "\nPrevious response: " + output + "\nCorrect it. " + envelope
            continue
        if log:
            log(attempt, output, "")
        return candidate
    return None
