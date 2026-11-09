"""Run the existing three-task HCL driver with a LangGraph answer backend.

Paths are supplied through environment variables; the launcher uses neutral argv.
Private data and run outputs never belong in the public source directory.
"""
import json
import os
from pathlib import Path
import re
import time
import traceback
from types import MethodType

from graph_agent import GraphAgentModel
from hcl.models import LocalTransformersModel, _state_phase
from hcl.pipeline import HCLPipeline
from hcl.runtime_fail_closed import install_memory_selector_fail_closed
from medical_hcl.bridge import construct_pipeline
from medical_hcl.data import StrictMedicalLoader


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def canonical_bbox(raw, question):
    # Preserve the old run's packaging normalization; no coordinate correction.
    if "output only one json object with bbox" not in question.lower():
        return raw
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        value = json.loads(fence.group(1) if fence else text)
    except ValueError:
        return raw
    item = value[0] if isinstance(value, list) and len(value) == 1 else value
    if isinstance(item, dict) and set(item).issubset({"bbox", "bbox_2d", "label"}):
        if ("bbox" in item) != ("bbox_2d" in item):
            box = item.get("bbox", item.get("bbox_2d"))
            if isinstance(box, list) and len(box) == 4:
                return json.dumps({"bbox": box})
    return raw


class BudgetExceeded(RuntimeError):
    pass


class AuditedBackend(LocalTransformersModel):
    def __init__(self, root, protocol, budget):
        super().__init__(name="Qwen2.5-VL-7B-Instruct", path=os.environ["MODEL_PATH"],
                         torch_dtype="bfloat16", attn_implementation="sdpa", temperature=0,
                         max_new_tokens=2048, answer_max_new_tokens=512,
                         enable_thinking=False, local_files_only=True)
        self.root, self.protocol, self.budget = root, protocol, budget
        self.calls = budget.get("initial_model_calls", 0)
        self.scope = "setup"

    def generate(self, prompt, *, state):
        if time.time() >= self.budget["deadline_epoch"] or (self.root / "STOPPED.json").exists():
            raise BudgetExceeded("original deadline or storage guard reached")
        if self.calls >= self.protocol["max_model_calls"]:
            raise BudgetExceeded("model call budget reached")
        self.calls += 1
        start = time.time()
        write_json(self.root / "PROGRESS.json", {"calls": self.calls, "scope": self.scope,
                                               "phase": _state_phase(state), "time": start})
        answer = None
        try:
            answer = super().generate(prompt, state=state)
            return canonical_bbox(answer, str(state.get("question", "")))
        finally:
            with (self.root / "private/model_calls.jsonl").open("a") as f:
                f.write(json.dumps({"call": self.calls, "scope": self.scope, "phase": _state_phase(state),
                                    "seconds": time.time() - start, "success": answer is not None,
                                    "prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")


def integrate(pipeline, backend, graph_model, protocol):
    # HCL metadata remains text/vision generation; only final answering runs the graph.
    pipeline.model = graph_model
    pipeline.selection_model = pipeline.judge_model = pipeline.memory_model = backend
    pipeline.memory.model = backend
    pipeline.optimizer.candidate_count = protocol["candidate_count_per_component"]
    pipeline.optimizer.min_historical_anchor_count = protocol["historical_anchor_minimum_total"]
    pipeline.optimizer.min_historical_anchors_per_task = protocol["historical_anchor_minimum_per_task"]
    pipeline.memory.config.anchor_capacity_per_task = protocol["samples_per_role_per_task"]
    pipeline.memory.anchor.capacity_per_task = protocol["samples_per_role_per_task"]
    pipeline.router_config.use_chat_tool_calls = False
    pipeline.router_config.tool_auto_select = False
    pipeline.router_config.workflow_selector_max_new_tokens = 512
    pipeline.router_config.skill_selector_max_new_tokens = 256
    pipeline.router_config.skill_selector_fallback_to_empty_on_invalid = True
    pipeline.router_config.memory_selector_max_new_tokens = 256
    pipeline.router_config.memory_selector_retry_max_new_tokens = 256
    pipeline.task_interface_config.max_new_tokens = 640
    pipeline.task_interface_config.retry_max_new_tokens = 1024
    pipeline.task_interface_config.fallback_to_rules_on_invalid = True
    pipeline.memory.config.abstract_max_new_tokens = 512
    pipeline.memory.abstract.max_new_tokens = 512
    install_memory_selector_fail_closed(HCLPipeline)

    def no_test_probe(self, *args, **kwargs):
        if self.forgetting_probe.enabled:
            raise ValueError("test probes are forbidden")
        return None
    pipeline._build_probe_set = MethodType(no_test_probe, pipeline)

    original_predict = pipeline._predict_with_router
    def scoped(self, *args, **kwargs):
        previous = backend.scope
        backend.scope = kwargs.get("progress_label", "predict")
        try:
            return original_predict(*args, **kwargs)
        finally:
            backend.scope = previous
    pipeline._predict_with_router = MethodType(scoped, pipeline)

    original_batch = pipeline._evaluate_train_batch
    def with_execution_feedback(self, *args, **kwargs):
        first = len(graph_model.episodes)
        result = original_batch(*args, **kwargs)
        training = graph_model.episodes[first:]
        counts = {}
        for episode in training:
            for event in episode["events"]:
                if event["node"] == "execute":
                    counts[event["tool"]] = counts.get(event["tool"], 0) + 1
        # Only this training batch contributes execution diagnostics to proposals.
        # These are observed process statistics, not causal failure attribution.
        result["metrics"]["graph_execution"] = {
            "episodes": len(training), "completed": sum(e["status"] == "completed" for e in training),
            "tool_counts": counts, "failures": [e.get("error") for e in training if e["status"] != "completed"]}
        return result
    pipeline._evaluate_train_batch = MethodType(with_execution_feedback, pipeline)


def main():
    root = Path(os.environ["RUN_ROOT"])
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    budget = json.loads((root / "budget.json").read_text())
    prepared = Path(os.environ["PREPARED_DATA"])
    private = root / "private"
    private.mkdir(exist_ok=True)
    write_json(root / "protocol.json", protocol)
    loader = StrictMedicalLoader(prepared)
    panel = {}
    groups = {}
    for task in protocol["tasks"]:
        panel[task] = {}
        for role in ("train", "validation", "anchor", "test"):
            rows = loader.load_task_spec({"task_name": task}, split=role,
                                        limit=protocol["samples_per_role_per_task"])
            panel[task][role] = [x["task_id"] for x in rows]
            for row in rows:
                group = row["metadata"]["group_id"]
                if group in groups and groups[group] != role:
                    raise ValueError("cross-role group leakage")
                groups[group] = role
                if not Path(row["image"]).is_file():
                    raise FileNotFoundError("selected image missing")
    write_json(private / "selected_panel.json", panel)
    backend = AuditedBackend(root, protocol, budget)
    backend._load()
    assert not backend._model.training and all(not p.requires_grad for p in backend._model.parameters())
    tool_runtime = None
    if protocol.get("enable_image_operations", False):
        from medical_transfer.six_tool_demo.vision_tools import VisionTools
        tool_runtime = VisionTools(private / "tool_artifacts")
    graph_model = GraphAgentModel(backend, private / "graph_traces.jsonl",
                                  max_tools=protocol["max_tool_calls_per_episode"], tool_runtime=tool_runtime)
    pipeline = construct_pipeline(prepared, private / "pipeline", model=backend,
                                  real_execution_authorized=True)
    integrate(pipeline, backend, graph_model, protocol)
    write_json(root / "MODEL_READY.json", {"time": time.time(), "frozen": True})
    count = protocol["samples_per_role_per_task"]
    if os.environ.get("RUN_MODE") == "interface_check":
        from medical_hcl.evaluation import score_case
        vocabulary = json.loads((prepared / "class_vocabulary.json").read_text())
        checks = []
        for task in protocol["tasks"]:
            examples = loader.load_task_spec({"task_name": task}, split="train", limit=1)
            before = len(graph_model.episodes)
            prediction = pipeline._predict(examples, progress_label="interface_check:" + task)[0]
            events = graph_model.episodes[before:]
            checks.append({"task": task, "graph_completed": len(events) == 1 and events[0]["status"] == "completed",
                           "format_valid": score_case(examples[0], prediction["answer"], vocabulary)[0],
                           "tools": [x["tool"] for e in events for x in e["events"] if x["node"] == "execute"]})
        write_json(root / "INTERFACE_CHECK.json", {"checks": checks, "model_calls": backend.calls,
            "passed": all(c["graph_completed"] and c["tools"] for c in checks),
            "correctness_not_a_launch_criterion": True})
        return
    result = pipeline.continual_train(prepared / "taskstream.json", train_limit_per_task=count,
                                      validation_limit_per_task=count, test_limit_per_task=count,
                                      batchsize=count)
    assert pipeline.historical_task_probe_sets == {}
    assert all(not p.requires_grad for p in backend._model.parameters())
    trace_summary = {
        "episodes": len(graph_model.episodes),
        "completed": sum(e["status"] == "completed" for e in graph_model.episodes),
        "failed": sum(e["status"] != "completed" for e in graph_model.episodes),
        "tool_calls": sum(sum(x["node"] == "execute" for x in e["events"]) for e in graph_model.episodes),
        "controller_calls": sum(sum(x["node"] == "process" for x in e["events"]) for e in graph_model.episodes),
    }
    summary = {"purpose": protocol["purpose"], "stages": len(result.continual_matrix["stages"]),
               "matrix": result.continual_matrix, "graph": trace_summary, "model_calls": backend.calls,
               "run_model_calls": backend.calls - budget.get("initial_model_calls", 0),
               "generated_candidates": len(result.generated_candidates),
               "accepted_prompt_updates": sum(bool(x.get("accepted")) for x in pipeline.optimizer.history),
               "memory": pipeline.memory.summary(), "model_frozen": True,
               "test_probes_disabled": True, "efficacy_established": False,
               "elapsed_seconds": time.time() - budget["started_epoch"]}
    write_json(root / "RESULT.json", summary)
    write_json(root / "FINISHED.json", {"time": time.time(), "stages": summary["stages"]})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        root = Path(os.environ["RUN_ROOT"])
        write_json(root / "FAILED.json", {"time": time.time(), "traceback": traceback.format_exc()})
        raise
