"""Exercise an actual upstream SkillOpt text patch inside a real HCL tool call."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
base = Path(os.environ["PILOT_ROOT"])
out = root / "run"
p = json.loads((root / "protocol.json").read_text())
design = json.loads((root / "design.json").read_text())
sys.path.insert(0, str(root))
budget = json.loads((root / "budget.json").read_text())
sys.path[:0] = [str(base / "code"), *p["python_paths"]]

from agent import GraphAgent, SkillExecutor
from core import decode, validate_library
from evaluate import score
from runtime import EpisodeTools, FrozenModels


def guard():
    if time.time() >= budget["soft_deadline_epoch"] or (root / "HARD_STOP.json").exists():
        raise RuntimeError("time or storage guard")
    if time.time() - json.loads((root / "resource_status.json").read_text())["time"] > 120:
        raise RuntimeError("stale guard")


def run():
    spec = importlib.util.spec_from_file_location("upstream_skill", root / "upstream" / "skill.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    state = json.loads((base / "run/state_T1.json").read_text())
    train = json.loads((base / "private_data.json").read_text())["train"]
    models = FrozenModels(p, out, guard)
    # Explicit one-tool engineering seed. Only the natural-language tool argument is editable.
    seed = "Locate the visible polyp region(s)."
    evidence = [{"tools": [t["tool"] for t in e["tool_trajectory"]], "score": e["feedback"]["score"],
                 "success": e["feedback"]["success"]} for e in state["raw"]["T1"][:4]]
    prompt = ("Improve the single localization instruction for a frozen image-reading tool, using these "
              "training outcomes. Suggest one short, reusable visual or procedural cue. No coordinates, "
              "historical answers or cases. Return one JSON patch with key edits containing one object. "
              "Use op=append and content=<one short sentence>, or op=replace with target=<exact current "
              "sentence> and content=<new instruction>. No code, markdown, other keys or explanations. "
              "Current instruction: " + seed + "\nTraining outcomes: " + json.dumps(evidence))
    typed_slot = design.get("patch_format") == "single-slot-v2"
    if typed_slot:
        prompt += '\nEditable document at initialization: ' + json.dumps([{"content": seed}])
        prompt += '\nThe only permitted JSON Patch path is /0/content (replace). An equivalent native edits envelope is also accepted.'
    attempts = []
    instruction = None
    models.phase = "native_training_proposal"
    for _ in range(2):
        original = models.vlm.config.system_prompt
        try:
            models.vlm.config.system_prompt = "Return a single JSON patch conforming exactly to the requested structure."
            output = models.generate(prompt, max_tokens=256)
        finally:
            models.vlm.config.system_prompt = original
        try:
            if typed_slot:
                from patch_format import normalize
                patch, representation = normalize(output, seed)
            else:
                patch, representation = decode(output), "native"
            if set(patch) != {"edits"} or not isinstance(patch["edits"], list) or len(patch["edits"]) != 1:
                raise ValueError("one edit required")
            edit = patch["edits"][0]
            op = edit["op"]
            if op not in ("append", "replace") or not isinstance(edit["content"], str) or not edit["content"].strip():
                raise ValueError("expected nonempty append or replace")
            updated, reports = upstream.apply_patch_with_report(seed, patch)
            if reports[0]["status"] not in ("applied_append", "applied_replace") or updated.strip() == seed:
                raise ValueError("patch did not change the instruction")
            if len(updated) > 600:
                raise ValueError("instruction exceeds the HCL limit")
            make_skill(updated)
            attempts.append({"output": output, "report": reports, "representation": representation})
            instruction = updated
            break
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            attempts.append({"output": output, "error": str(exc)})
            prompt += "\nError: " + str(exc) + ". Correct the previous JSON: " + output
    (out / "private_native_edits.json").write_text(json.dumps(attempts))
    summary = {"upstream_commit": "dc0b5e654cafa1073a2819086151c1e0d13012f0", "patched": instruction is not None,
               "attempts": len(attempts), "case_role": "fifth previously assigned training image",
               "official_function": "skillopt.optimizer.skill.apply_patch_with_report"}
    if instruction is not None:
        case = train[4]
        graph = GraphAgent(models, p["runtime"])
        for arm, text in (("seed_direct", seed), ("patched_direct", instruction), ("patched_graph", instruction)):
            models.phase = arm
            skill = make_skill(text)
            tools = EpisodeTools(models, case["image"], out / "private_artifacts" / arm, p["runtime"])
            if arm == "patched_graph":
                episode = graph.run("T1", tools, [skill], [])
            else:
                result = SkillExecutor(tools).run(skill)
                episode = {"answer": result.get("proposed_answer", {}), "error": "" if result["status"] == "ok" else "tool_failure",
                           "skills_used": [skill["name"]]}
            metric = score("T1", case, episode, tools, p["evaluation"])
            (out / ("private_" + arm + ".json")).write_text(json.dumps({"tools": tools.events, "episode": episode}))
            summary[arm] = {"valid": metric["valid"], "primitive_calls": len(tools.events),
                            "skill_called": skill["name"] in episode["skills_used"],
                            "instruction_reached_tool": any(t["tool"] == "localize" and t["args"].get("instruction") == text for t in tools.events)}
        summary["representation"] = attempts[-1]["representation"]
        summary["edit_status"] = attempts[-1]["report"][0]["status"]
    models.assert_frozen()
    summary.update(vlm_calls=models.calls, sam_calls=models.sam_calls)
    (out / "result.json").write_text(json.dumps(summary, indent=2))


def make_skill(text):
    return validate_library([{"name": "locate_region", "task": "T1", "description": "Localize a visible polyp.",
         "applies_when": "A box is requested for the visible polyp.", "steps": [
         {"id": "loc", "tool": "localize", "args": {"instruction": text}}], "output": {"bbox": "$loc.bbox"}}])[0]


if __name__ == "__main__":
    run()
