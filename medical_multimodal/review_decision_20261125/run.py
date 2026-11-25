import json
import os
import sys
import time
from pathlib import Path

from PIL import ImageDraw

ROOT = Path(os.environ["RUN_ROOT"])
BASE = Path(os.environ["PILOT_ROOT"])
OUT = ROOT / "run"
P = json.loads((ROOT / "protocol.json").read_text())
B = json.loads((ROOT / "budget.json").read_text())
sys.path[:0] = [str(ROOT), str(BASE / "code"), *P["python_paths"]]
from decision import summarize, training_summary
from evaluate import score
from runtime import EpisodeTools, FrozenModels


def guard():
    status = json.loads((ROOT / "resource_status.json").read_text())
    if time.time() >= B["soft_deadline_epoch"] or time.time() - status["time"] > 120 or (ROOT / "HARD_STOP.json").exists():
        raise RuntimeError("resource guard")


def observe(tools, box):
    x1, y1, x2, y2 = box
    image = tools.image.copy()
    ImageDraw.Draw(image).rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 240, 255), width=2)
    path = tools.out / "first_box.png"
    image.save(path)
    prompt = ("The cyan rectangle is a proposed localization on this current image. Judge how it fits the visible polyp. "
              "Return exactly one word: TIGHT if it fits, LOOSE if it encloses much excess tissue, CLIPPED if part of the "
              "polyp extends outside it, MISSED if the target is elsewhere, UNSURE if unclear.")
    reply = tools.models.generate(prompt, str(path), 32).strip().upper().strip(" .\"'`\n")
    allowed = ("TIGHT", "LOOSE", "CLIPPED", "MISSED", "UNSURE")
    return {"area": (x2-x1)*(y2-y1)/(tools.w*tools.h),
            "margin": min(x1/tools.w,y1/tools.h,(tools.w-x2)/tools.w,(tools.h-y2)/tools.h),
            "visual": reply if reply in allowed else "UNPARSED"}


def paired(models, cases, role):
    rows = []
    for i, case in enumerate(cases):
        guard()
        models.phase, models.scope = role, str(i)
        tools = EpisodeTools(models, case["image"], OUT / "private_artifacts" / role / str(i), P["runtime"])
        first = tools.execute("localize", {})
        cues = {}
        reviewed = first
        review_calls = 0
        if first["status"] == "ok":
            cues = observe(tools, first["bbox"])
            before = models.calls
            roi = tools.execute("expand_box", {"bbox": first["bbox"], "fraction": .15})
            reviewed = tools.execute("localize", {"roi": roi["bbox"]}) if roi["status"] == "ok" else roi
            review_calls = models.calls - before
        # Labels enter here, after the two action outcomes and decision cues are fixed.
        def grade(observation):
            answer = {"bbox": observation["bbox"]} if observation["status"] == "ok" else {}
            return score("T1", case, {"answer": answer}, tools, P["evaluation"])
        a, b = grade(first), grade(reviewed)
        row = {"first_valid": a["valid"], "review_valid": b["valid"], "stop": a["score"],
               "review": b["score"], "review_calls": review_calls, "cues": cues}
        rows.append(row)
        with (OUT / ("private_" + role + ".jsonl")).open("a") as f:
            f.write(json.dumps({"id":case["id"], **row, "tool_events":tools.events}) + "\n")
    models.assert_frozen()
    return rows


def main():
    data = json.loads((BASE / "private_data.json").read_text())
    train = data["train"]
    dev = data["validation"]
    groups = {c["group_id"] for c in train + dev}
    if len(train) != 12 or len(dev) != 12 or len(groups) != 24:
        raise ValueError("expected disjoint, unique original source groups")
    models = FrozenModels(P, OUT, guard)
    t = training_summary(paired(models, train, "train"))
    (OUT / "training_decision.json").write_text(json.dumps(t, indent=2))
    result = {"training": t, "development": None}
    if t["open_development"]:
        d = summarize(paired(models, dev, "development"))
        m = d["policies"][t["frozen_policy"]]
        fixed = [d["policies"][p] for p in ("stop", "review")]
        d["frozen_meets_criterion"] = (0 < m["reviews"] < len(dev)
            and m["mean_iou"] >= max(x["mean_iou"] for x in fixed) + .02
            and m["success"] >= max(x["success"] for x in fixed)
            and m["mean_vlm_calls"] <= fixed[1]["mean_vlm_calls"])
        result["development"] = d
    result.update(vlm_calls=models.calls, sam_calls=models.sam_calls)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
