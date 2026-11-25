"""Compare two frozen tool instructions via the ordinary HCL graph on development data."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
base = Path(os.environ["PILOT_ROOT"])
out = root / "run"
config = json.loads((root / "protocol.json").read_text())
budget = json.loads((root / "budget.json").read_text())
sys.path[:0] = [str(root), str(base / "code"), *config["python_paths"]]

from agent import GraphAgent
from evaluate import aggregate, score
from guidance import GuidedEpisodeTools
from patch_format import normalize
from runtime import FrozenModels


def guard():
    status = json.loads((root / "resource_status.json").read_text())
    if time.time() >= budget["soft_deadline_epoch"] or time.time() - status["time"] > 120 or (root / "HARD_STOP.json").exists():
        raise RuntimeError("resource guard")


def run():
    parent = Path(os.environ["PATCH_RUN_ROOT"]) / "run"
    result = json.loads((parent / "result.json").read_text())
    assert result["patched"] and result["patched_direct"]["valid"]
    spec = importlib.util.spec_from_file_location("upstream_skill", root / "upstream" / "skill.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    seed = "Locate the visible polyp region(s)."
    receipt = json.loads((parent / "private_native_edits.json").read_text())[-1]
    patch, _ = normalize(receipt["output"], seed)
    edited, reports = upstream.apply_patch_with_report(seed, patch)
    assert reports == receipt["report"] and edited != seed
    data = json.loads((base / "private_data.json").read_text())
    cases = data["validation"][:4]
    assert len({c["group_id"] for c in cases}) == 4
    assert not {c["group_id"] for c in cases} & {c["group_id"] for c in data["train"]}
    models = FrozenModels(config, out, guard)
    graph = GraphAgent(models, config["runtime"])
    arms, paired = {}, {}
    for label, instruction in (("seed", seed), ("edited", edited)):
        rows = []
        delivered = 0
        exposed = 0
        for i, case in enumerate(cases):
            models.phase = label
            models.scope = str(i)
            tools = GuidedEpisodeTools(models, case["image"], out / "private_artifacts" / label / str(i),
                                       config["runtime"], guidance=instruction)
            episode = graph.run("T1", tools, [], [])
            metrics = score("T1", case, episode, tools, config["evaluation"])
            count = sum(t["source"] == "controller" and t["tool"] == "localize"
                        and t["args"].get("instruction", "").endswith(instruction.strip())
                        and t["result"]["status"] == "ok" for t in tools.events)
            delivered += count
            exposed += int(count > 0)
            rows.append({"id": case["id"], "metrics": metrics, "episode": episode})
        models.assert_frozen()
        (out / ("private_" + label + ".json")).write_text(json.dumps(rows))
        arms[label] = {**aggregate(rows), "successful_guided_calls": delivered, "exposed_episodes": exposed}
        paired[label] = [r["metrics"]["success"] for r in rows]
    summary = {"arms": arms, "gains": sum(b and not a for a,b in zip(paired["seed"], paired["edited"])),
               "losses": sum(a and not b for a,b in zip(paired["seed"], paired["edited"])),
               "vlm_calls": models.calls, "sam_calls": models.sam_calls, "candidate_committed": False}
    (out / "result.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
