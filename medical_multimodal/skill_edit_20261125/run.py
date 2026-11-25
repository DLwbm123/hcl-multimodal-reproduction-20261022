"""Run the declared development comparison against the original frozen HCL runtime."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["RUN_ROOT"])
BASE = Path(os.environ["PILOT_ROOT"])
OUT = ROOT / "run"
OUT.mkdir(exist_ok=True)
P = json.loads((ROOT / "protocol.json").read_text())
B = json.loads((ROOT / "budget.json").read_text())
DESIGN = json.loads((ROOT / "design.json").read_text())
sys.path[:0] = [str(ROOT), str(BASE / "code"), *P["python_paths"]]

from agent import GraphAgent, SkillExecutor
from core import validate_library
from evaluate import accept, aggregate, score
from proposal import propose
from runtime import EpisodeTools, FrozenModels


def save(name, data):
    path = OUT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def guard():
    stamp = json.loads((ROOT / "resource_status.json").read_text())["time"]
    if time.time() - stamp > 120 or time.time() >= B["soft_deadline_epoch"] or (ROOT / "HARD_STOP.json").exists():
        raise RuntimeError("resource or time guard stopped the run")


def run():
    guard()
    data = json.loads((BASE / "private_data.json").read_text())
    train, validation = data["train"], data["validation"][:4]
    if len(validation) != 4 or len({c["group_id"] for c in validation}) != 4 or len(train) < 5:
        raise ValueError("insufficient source groups")
    if {c["group_id"] for c in train} & {c["group_id"] for c in validation}:
        raise ValueError("training and development overlap")
    models = FrozenModels(P, OUT, guard)
    graph = GraphAgent(models, P["runtime"])
    committed = {arm: [] for arm in DESIGN["arms"]}
    rows_cache = {}
    events = []
    schema = {}

    def proposal(task, arm, evidence, draft):
        attempts = []

        def log(index, output, error):
            attempts.append({"attempt": index + 1, "error": error, "output": output})

        models.phase = "proposal"
        models.scope = task + "_" + arm
        def generate(prompt):
            original = models.vlm.config.system_prompt
            try:
                models.vlm.config.system_prompt = "Design reusable procedures from training feedback. Output exactly the JSON object specified by the user."
                return models.generate(prompt, max_tokens=P["runtime"]["proposal_max_tokens"])
            finally:
                models.vlm.config.system_prompt = original

        skill = propose(generate, task, evidence, draft, "edit" if arm == "edit" else "full", log)
        save("private_proposal_" + models.scope + ".json", attempts)
        schema[models.scope] = {"valid": skill is not None, "attempts": len(attempts),
                                "failed_attempts": sum(bool(a["error"]) for a in attempts)}
        return skill

    def evaluate(task, library, label):
        key = (task, json.dumps(library, sort_keys=True))
        if key in rows_cache:
            return rows_cache[key]
        rows = []
        models.phase = "development"
        for i, case in enumerate(validation):
            guard()
            models.scope = f"{label}_{task}_{i}"
            tools = EpisodeTools(models, case["image"], OUT / "private_artifacts" / models.scope, P["runtime"])
            episode = graph.run(task, tools, validate_library(library), [])
            rows.append({"id": case["id"], "metrics": score(task, case, episode, tools, P["evaluation"]), "episode": episode})
        models.assert_frozen()
        save("private_" + label + "_" + task + ".json", rows)
        rows_cache[key] = rows
        return rows

    for task in ("T1", "T2"):
        state = json.loads((BASE / "run" / f"state_{task}.json").read_text())
        evidence = state["raw"][task][:4]
        if len(evidence) != 4:
            raise ValueError("missing training trajectories")
        shared = proposal(task, "seed", evidence, None)
        # Freeze both candidates before either is evaluated.
        candidates = {arm: proposal(task, arm, evidence, shared) for arm in committed}
        for arm, skill in candidates.items():
            guard()
            event = {"task": task, "arm": arm, "accepted": False, "schema_valid": skill is not None}
            events.append(event)
            if skill is None:
                event["reason"] = "invalid_proposal"
                continue
            library = committed[arm]
            if any(s["name"] == skill["name"] for s in library):
                event["reason"] = "name_conflicts_with_committed_skill"
                continue
            trial = validate_library([*library, skill])
            models.phase = "training_probe"
            models.scope = task + "_" + arm
            case = train[4]
            tools = EpisodeTools(models, case["image"], OUT / "private_probe" / models.scope, P["runtime"])
            try:
                result = SkillExecutor(tools).run(skill)
                probe = {"answer": result.get("proposed_answer", {}), "error": "" if result["status"] == "ok" else "tool_failure"}
            except (KeyError, ValueError, TypeError) as exc:
                probe = {"answer": {}, "error": str(exc)}
            valid = score(task, case, probe, tools, P["evaluation"])["valid"]
            save("private_probe_" + models.scope + ".json", {"skill": skill, "tools": tools.events, "probe": probe})
            event.update(probe_valid=valid, probe_tool_calls=len(tools.events))
            if not valid:
                event["reason"] = "invalid_training_execution"
                continue
            baseline = evaluate(task, library, arm + "_before")
            candidate = evaluate(task, trial, arm + "_after")
            history_before = evaluate("T1", library, arm + "_history_before") if task == "T2" else []
            history_after = evaluate("T1", trial, arm + "_history_after") if task == "T2" else []
            gate = accept(baseline, candidate, history_before, history_after, P["gate"])
            use = sum(skill["name"] in row["episode"]["skills_used"] for row in candidate)
            if not use:
                gate["accepted"] = False
                gate["reasons"].append("new_skill_never_called")
            event.update(gate=gate, voluntary_new_skill_episodes=use, accepted=gate["accepted"])
            if gate["accepted"]:
                committed[arm] = trial
        save("progress.json", {"finished_stage": task, "proposals": schema, "events": events})
        save("private_committed_" + task + ".json", committed)
    models.assert_frozen()
    save("result.json", {"design": DESIGN["purpose"], "proposals": schema, "events": events,
                         "committed_skill_counts": {arm: len(lib) for arm, lib in committed.items()},
                         "vlm_calls": models.calls, "sam_calls": models.sam_calls})


if __name__ == "__main__":
    run()
