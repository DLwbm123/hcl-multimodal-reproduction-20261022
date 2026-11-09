"""CPU contract checks. Scripted outputs are fixtures, never medical evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from graph_agent import AgentProtocolError, GraphAgentModel, decode_action
from hcl.models import RuleBasedSmokeModel
from medical_hcl.bridge import construct_pipeline
from run import integrate


class FixtureBackend(RuleBasedSmokeModel):
    scope = "fixture"

    def __init__(self):
        self.inputs = []

    def generate(self, prompt, *, state):
        self.inputs.append((prompt, state))
        phase = state.get("router_phase")
        if phase in {"graph_controller", "graph_tool"}:
            question = state["question"]
            if "Classify" in question:
                tool, answer = "classify_image", "normal"
            elif "bbox" in question:
                tool, answer = "localize_region", '{"bbox":[1,1,8,8]}'
            else:
                tool, answer = "answer_visual_question", "one"
            if phase == "graph_tool":
                return answer
            if "Tool observations:\n[]" in prompt:
                return json.dumps({"tool": tool, "instruction": "Follow the current question."})
            return json.dumps({"answer": answer})
        return super().generate(prompt, state=state)


class GraphChecks(unittest.TestCase):
    def test_real_graph_tool_message_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FixtureBackend()
            model = GraphAgentModel(backend, Path(directory) / "trace.jsonl")
            answer = model.generate("Classify this image.", state={"question": "Classify this image.",
                "task_context": {"files": []}, "answer": "SECRET_LABEL", "task_name": "SECRET_STAGE"})
            self.assertEqual(answer, "normal")
            episode = model.episodes[-1]
            self.assertEqual([x["node"] for x in episode["events"]], ["process", "execute", "process"])
            self.assertGreater(episode["checkpoint_count"], 2)
            for prompt, state in backend.inputs:
                self.assertNotIn("SECRET_LABEL", prompt)
                self.assertNotIn("SECRET_STAGE", prompt)
                self.assertNotIn("answer", state)
                self.assertNotIn("task_name", state)

    def test_invalid_actions_are_failed_samples(self):
        class Invalid(FixtureBackend):
            def generate(self, prompt, *, state):
                return '{"tool":"shell","instruction":"anything"}'
        with tempfile.TemporaryDirectory() as directory:
            model = GraphAgentModel(Invalid(), Path(directory) / "trace.jsonl")
            self.assertEqual(model.generate("q", state={"question": "q"}), "")
            self.assertEqual(model.episodes[-1]["status"], "failed")

    def test_repeated_tool_calls_are_bounded(self):
        class Loop(FixtureBackend):
            def generate(self, prompt, *, state):
                return ('{"tool":"answer_visual_question","instruction":"inspect"}'
                        if state["router_phase"] == "graph_controller" else "observation")
        with tempfile.TemporaryDirectory() as directory:
            model = GraphAgentModel(Loop(), Path(directory) / "trace.jsonl", max_tools=2)
            self.assertEqual(model.generate("q", state={"question": "q"}), "")
            self.assertEqual(sum(e["node"] == "execute" for e in model.episodes[-1]["events"]), 2)

    def test_budget_errors_abort_instead_of_becoming_success(self):
        class Exhausted(FixtureBackend):
            def generate(self, prompt, *, state):
                raise RuntimeError("budget exhausted")
        with tempfile.TemporaryDirectory() as directory:
            model = GraphAgentModel(Exhausted(), Path(directory) / "trace.jsonl")
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                model.generate("q", state={"question": "q"})
            self.assertEqual(model.episodes[-1]["status"], "aborted")

    def test_reject_extra_tool_arguments(self):
        with self.assertRaises(AgentProtocolError):
            decode_action('{"tool":"classify_image","instruction":"x","image_path":"/other"}')
        with self.assertRaises(AgentProtocolError):
            decode_action('{"tool":[],"instruction":"x"}')

    def test_native_bbox_answer_after_tool(self):
        class Native(FixtureBackend):
            def generate(self, prompt, *, state):
                if state.get("router_phase") == "graph_controller" and "Tool observations:\n[]" not in prompt:
                    return '{"bbox":[1,1,8,8]}'
                return super().generate(prompt, state=state)
        with tempfile.TemporaryDirectory() as directory:
            model = GraphAgentModel(Native(), Path(directory) / "trace.jsonl")
            self.assertEqual(model.generate("bbox", state={"question": "bbox"}), '{"bbox":[1,1,8,8]}')
            self.assertEqual(model.episodes[-1]["status"], "completed")

    def test_full_three_stage_pipeline_with_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "FROZEN.json").write_text('{"seed":20261022}')
            (prepared / "class_vocabulary.json").write_text('["normal","abnormal"]')
            protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
            specs = []
            for task, question, answer in [
                ("classification", "Classify this image; normal or abnormal.", "normal"),
                ("polyp_localization", "Output only one JSON object with bbox.", {"bbox": [1, 1, 8, 8]}),
                ("visual_qa", "How many objects are visible?", "one")]:
                specs.append({"task_name": task, "task_type": "medical_" + task})
                for role in ("h_search", "validation", "retention_anchor", "test"):
                    key = task + "_" + role
                    visible = {"id": key, "question": question, "visible_context": {"width": 10, "height": 10}}
                    private = {"id": key, "answer": answer, "metadata": {
                        "role": role, "group_id": key, "pixel_id": key, "width": 10, "height": 10,
                        "question_type": "text", "answer_kind": "text"}}
                    for kind, row in (("visible", visible), ("private", private)):
                        path = prepared / kind / task / (role + ".jsonl")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(row) + "\n")
            stream = prepared / "taskstream.json"
            stream.write_text(json.dumps({"tasks": specs}))
            backend = FixtureBackend()
            graph = GraphAgentModel(backend, root / "traces.jsonl")
            pipeline = construct_pipeline(prepared, root / "pipeline", model=backend,
                                          real_execution_authorized=True)
            integrate(pipeline, backend, graph, protocol)
            result = pipeline.continual_train(stream, train_limit_per_task=1,
                validation_limit_per_task=1, test_limit_per_task=1, batchsize=1)
            self.assertEqual(len(result.continual_matrix["stages"]), 3)
            self.assertEqual(pipeline.historical_task_probe_sets, {})
            self.assertTrue(graph.episodes)
            self.assertTrue(all(e["status"] == "completed" for e in graph.episodes))
            self.assertEqual(len(result.generated_candidates), 12)
            self.assertEqual(pipeline.memory.summary()["raw"], 0)


if __name__ == "__main__":
    unittest.main()
