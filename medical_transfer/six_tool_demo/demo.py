"""One real-model coverage episode; all paths are supplied through environment."""
import json
import os
from pathlib import Path
import time
import traceback

from PIL import Image
import numpy as np

from graph_agent import GraphAgentModel
from hcl.models import LocalTransformersModel
from vision_tools import VisionTools


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    root = Path(os.environ["RUN_ROOT"])
    protocol = json.loads((root / "protocol.json").read_text())
    budget = json.loads((root / "budget.json").read_text())
    prepared = Path(os.environ["PREPARED_ROOT"])
    # Do not use the evaluator/loader: only this visible training row is read.
    with (prepared / "visible/polyp_localization/h_search.jsonl").open() as f:
        visible = json.loads(next(f))
    image_path = visible["image"]
    with Image.open(image_path) as image:
        w, h = image.size
    bbox = [w//10, h//10, w-w//10, h-h//10]
    vocabulary = json.loads((prepared / "class_vocabulary.json").read_text())
    private = root / "private"
    private.mkdir(exist_ok=True)
    save(private / "input.json", {"visible": visible, "roi": bbox, "source_role": "h_search"})
    class Backend(LocalTransformersModel):
        scope = "six_tool_coverage"
        def __init__(self):
            super().__init__(name="Qwen2.5-VL-7B-Instruct", path=os.environ["MODEL_PATH"],
                             torch_dtype="bfloat16", attn_implementation="sdpa", temperature=0,
                             max_new_tokens=384, answer_max_new_tokens=192, local_files_only=True)
            self.calls = budget.get("initial_model_calls", 0)
        def generate(self, prompt, *, state):
            if time.time() >= budget["deadline_epoch"] or self.calls >= protocol["max_model_calls"]:
                raise RuntimeError("demo time/call budget exhausted")
            self.calls += 1
            save(root / "PROGRESS.json", {"calls": self.calls, "phase": state.get("router_phase"), "time": time.time()})
            state = {**state, "max_new_tokens": 384 if state.get("router_phase") == "graph_controller" else 192}
            output = super().generate(prompt, state=state)
            with (private / "model_calls.jsonl").open("a") as f:
                f.write(json.dumps({"phase": state["router_phase"], "prompt": prompt, "output": output,
                                    "files": state.get("task_context", {}).get("files", [])}) + "\n")
            return output
    backend = Backend()
    runtime = VisionTools(private / "artifacts")
    plan = [
        {"tool": "analyze_image_quality", "arguments": {}},
        {"tool": "classify_image", "arguments": {"instruction": "Choose one category from: " + str(vocabulary)}},
        {"tool": "localize_region", "arguments": {"instruction": "Locate a candidate polyp and return an original-pixel bbox."}},
        {"tool": "inspect_region", "arguments": {"bbox": bbox, "question": "Describe visible structures in this crop."}},
        {"tool": "segmentation", "arguments": {"bbox": bbox}},
        {"tool": "answer_visual_question", "arguments": {"instruction": "What is visible in this endoscopy image?"}},
    ]
    graph = GraphAgentModel(backend, private / "trace.jsonl", max_tools=protocol["max_tool_calls"],
                            tool_runtime=runtime, coverage_plan=plan)
    question = (
        "This is a prescribed ENGINEERING TOOL COVERAGE demo on one endoscopy image, not a diagnosis. "
        "Execute each of these tools once in this exact order: analyze_image_quality, classify_image, "
        "localize_region, inspect_region, segmentation, answer_visual_question. "
        "At each controller step choose the next unused tool. Do not give a final answer until all six returned. "
        "For analyze_image_quality use arguments {}. For classify_image request one category from " + str(vocabulary) + ". "
        "For localize_region request a candidate polyp bbox in original pixels. "
        "For BOTH inspect_region and segmentation use this FIXED demonstration ROI " + str(bbox) +
        "; this ROI is not a known lesion boundary. For inspect_region ask: Describe visible structures in this crop. "
        "For answer_visual_question ask: What is visible in this endoscopy image? "
        "After all six observations, summarize the tool execution and uncertainty briefly. "
        "Original image width=" + str(w) + ", height=" + str(h) + ".")
    answer = graph.generate(question, state={"question": question,
        "task_context": {"files": [{"type": "image", "path": image_path}]}})
    episode = graph.episodes[-1]
    events = [e for e in episode["events"] if e["node"] == "execute"]
    receipts = {e["tool"]: json.loads(e["result"]) for e in events if e["tool"] in runtime.descriptions}
    checks = {"graph_completed": episode["status"] == "completed",
              "all_six_tools": set(e["tool"] for e in events) == set(protocol["tools"]),
              "new_tool_statuses_ok": len(receipts) == 3 and all(r["status"] == "ok" for r in receipts.values())}
    if checks["new_tool_statuses_ok"]:
        mask = np.asarray(Image.open(runtime.output_dir / receipts["segmentation"]["mask_artifact"]))
        crop = Image.open(runtime.output_dir / receipts["inspect_region"]["artifact"])
        checks["mask_original_shape_binary"] = mask.shape == (h, w) and set(np.unique(mask)).issubset({0, 255})
        checks["crop_readable_expected_size"] = crop.size == (bbox[2]-bbox[0], bbox[3]-bbox[1])
        crop.close()
    frozen = not backend._model.training and all(not p.requires_grad for p in backend._model.parameters())
    checks["model_frozen"] = frozen
    save(private / "final.json", {"answer": answer, "receipts": receipts})
    save(root / "RESULT.json", {"passed": all(checks.values()), "checks": checks,
        "tool_sequence": [e["tool"] for e in events], "tool_calls": len(events),
        "model_calls": backend.calls, "controller_calls": episode.get("controller_calls"),
        "run_model_calls": backend.calls-budget.get("initial_model_calls", 0),
        "planned_actions": sum(e.get("decision_source") == "prescribed_plan" for e in episode["events"]),
        "model_controller_calls": sum(e.get("decision_source") == "model" for e in episode["events"]),
        "foreground_empty": receipts.get("segmentation", {}).get("empty_foreground"),
        "elapsed_seconds": time.time()-budget["started_epoch"],
        "scope": "prescribed coverage only; no medical accuracy, learning or autonomous routing claim"})
    if not all(checks.values()):
        raise RuntimeError("tool coverage acceptance failed; inspect preserved trace")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        root = Path(os.environ["RUN_ROOT"])
        save(root / "FAILED.json", {"traceback": traceback.format_exc(), "time": time.time()})
        raise
