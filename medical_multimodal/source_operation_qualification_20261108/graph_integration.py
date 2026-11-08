"""Native MedRAX graph integration with a scripted controller, not an efficacy trial."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from qualified_tool import install_shared_guard
from native_runtime import BoundedTool


def main():
    r = Path(os.environ["DIAGNOSTIC_ROOT"])
    cfg = json.loads((r / "config.private.json").read_text())
    manifest = json.loads((r / "manifest.private.json").read_text())
    rows, events, fatal = [], [], None
    def save(name, value):
        p = r / name
        t = p.with_suffix(".tmp")
        t.write_text(json.dumps(value, indent=2) + "\n")
        t.replace(p)
    def record(kind, value):
        events.append({"kind": kind, **value})
        save("events.private.json", events)
    def guard():
        if time.time() >= cfg["deadline"]:
            raise TimeoutError("Original deadline reached")
    save("STARTED.json", {"time": time.time(), "pid": os.getpid()})
    try:
        import torch
        import transformers
        from transformers import dynamic_module_utils
        from transformers.cache_utils import DynamicCache
        assert len(manifest) == 2 and all(x["privacy_approved"] for x in manifest)
        original_imports = dynamic_module_utils.get_imports
        def imports(filename):
            names = original_imports(filename)
            return [x for x in names if x != "tensorflow"] if Path(filename).name == "tokenization_chexagent.py" else names
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda cache: cache.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
        spec = importlib.util.spec_from_file_location("pinned_tool", cfg["upstream_tool"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        version = transformers.__version__
        try:
            dynamic_module_utils.get_imports = imports
            tool = module.XRayVQATool(model_name=cfg["model_root"], device="cuda:0", cache_dir=cfg["cache"])
        finally:
            dynamic_module_utils.get_imports, transformers.__version__ = original_imports, version
        context = tool.model.config.max_position_embeddings
        save("LOADED.json", {"time": time.time(), "transformers": version, "torch": torch.__version__, "context": context})
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.checkpoint.memory import MemorySaver
        spec = importlib.util.spec_from_file_location("native_graph", cfg["upstream_agent"])
        graph_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_module)
        state = {"calls": 0, "max_calls": 1, "images": set()}
        tools = [BoundedTool(tool, state, record, guard)]
        install_shared_guard(tools, state, record)
        paths = [x["path"] for x in manifest]
        cases = [
            ("single_0", [paths[0]], [paths[0]], True),
            ("single_1", [paths[1]], [paths[1]], True),
            ("over_window", paths, paths, False),
            ("outside_case", [paths[1]], [paths[0]], False),
        ]
        class ScriptedController:
            def __init__(self, requested):
                self.requested, self.calls, self.received = requested, 0, None
            def bind_tools(self, available):
                assert len(available) == 1 and available[0].name == "chest_xray_expert"
                return self
            def invoke(self, messages):
                guard()
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(content="", tool_calls=[{
                        "name": "chest_xray_expert", "id": "probe",
                        "args": {"image_paths": self.requested,
                                 "prompt": "Describe the chest X-ray findings.", "max_new_tokens": 256}}])
                assert self.calls == 2 and messages[-1].type == "tool"
                self.received = messages[-1].content
                return AIMessage(content="Tool result received; clinical correctness is not established.")
        for name, requested, allowed, expected_success in cases:
            guard()
            state.update(calls=0, images=set(allowed))
            before = len(events)
            controller = ScriptedController(requested)
            agent = graph_module.Agent(controller, tools, checkpointer=MemorySaver(), log_tools=False)
            output = agent.workflow.invoke({"messages": [HumanMessage(content="Run the fixed integration probe.")]},
                                          {"configurable": {"thread_id": name}, "recursion_limit": 6})
            calls = events[before:]
            q = [x for x in calls if x["kind"] == "expert_qualification"]
            t = [x for x in calls if x["kind"] == "tool_finished"]
            completed = len(t) == 1 and t[0]["status"] == "completed"
            generated = sum(x["generation_attempted"] for x in q)
            row = {"case": name, "expected_success": expected_success,
                   "tool_completed": completed, "generation_attempts": generated,
                   "scripted_controller_steps": controller.calls,
                   "tool_message_received": controller.received is not None,
                   "visual_calls": sum(x.get("visual_observation", {}).get("visual_calls", 0) for x in q),
                   "graph_returned": output["messages"][-1].type == "ai"}
            row["pass"] = (completed == expected_success and generated == int(expected_success)
                           and row["visual_calls"] == int(expected_success)
                           and row["tool_message_received"] and row["graph_returned"])
            rows.append(row)
            save("PROGRESS.json", {"recorded_cells": len(rows), "planned_cells": 4})
            save("results.json", rows)
    except Exception as exc:
        fatal = repr(exc)
        save("failure.private.json", {"error": fatal, "traceback": traceback.format_exc()})
    summary = {"finished_at": time.time(), "planned_cells": 4, "recorded_cells": len(rows),
               "passed_cells": sum(x["pass"] for x in rows), "fatal": fatal,
               "generation_attempts": sum(x["generation_attempts"] for x in rows),
               "controller_model_calls": 0, "controller_is_scripted": True,
               "clinical_accuracy_evaluated": False, "learning_evaluated": False,
               "all_six_tools_loaded": False}
    save("FINISHED.json", summary)
    return 0 if summary["passed_cells"] == 4 and fatal is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
