"""Synthetic contract checks, never medical accuracy evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from graph_agent import AgentProtocolError, GraphAgentModel, decode_action
from vision_tools import VisionTools


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = Image.new("RGB", (120, 80), (20, 20, 20))
        ImageDraw.Draw(image).ellipse((35, 20, 85, 60), fill=(220, 60, 60))
        self.path = self.root / "input.png"
        image.save(self.path)
        self.state = {"question": "demo", "task_context": {"files": [{"type": "image", "path": str(self.path)}]},
                      "answer": "PRIVATE_GOLD"}
        self.tools = VisionTools(self.root / "artifacts")

    def test_statistics_and_mask(self):
        quality = self.tools.execute("analyze_image_quality", {}, self.state, None)
        self.assertGreater(quality["laplacian_variance"], 0)
        result = self.tools.execute("segmentation", {"bbox": [25, 10, 95, 70]}, self.state, None)
        self.assertEqual(result["status"], "ok")
        mask = np.asarray(Image.open(self.tools.output_dir / result["mask_artifact"]))
        self.assertEqual(mask.shape, (80, 120))
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertEqual(int((mask > 0).sum()), result["foreground_pixels"])
        self.assertGreater(mask[40, 60], 0)
        self.assertEqual(mask[0, 0], 0)

    def test_crop_pixels_and_private_state_exclusion(self):
        outer = self
        class Backend:
            def generate(self, prompt, *, state):
                outer.assertNotIn("answer", state)
                outer.assertNotIn("PRIVATE_GOLD", prompt)
                crop = np.asarray(Image.open(state["task_context"]["files"][0]["path"]))
                original = np.asarray(Image.open(outer.path))
                np.testing.assert_array_equal(crop, original[10:70, 25:95])
                return "synthetic red object"
        result = self.tools.execute("inspect_region", {"bbox": [25, 10, 95, 70], "question": "Describe."},
                                    self.state, Backend())
        self.assertEqual(result["crop_size"], [70, 60])

    def test_fail_closed_arguments_and_bounds(self):
        for args in [{"bbox": [True, 0, 8, 8]}, {"bbox": [0, 0, 8, 8], "path": "/other"},
                     {"bbox": [0, 0, float("nan"), 8]}]:
            with self.assertRaises(AgentProtocolError):
                decode_action(json.dumps({"tool": "segmentation", "arguments": args}), self.tools)
        for bbox in [[0, 0, 999, 70], [0, 0, 120, 80]]:
            result = self.tools.execute("segmentation", {"bbox": bbox}, self.state, None)
            self.assertEqual(result["status"], "error")
        with self.assertRaises(AgentProtocolError):
            decode_action('{"tool":"classify_image","arguments":{"category":"polyps"}}', self.tools)
        action = decode_action('{"tool":"classify_image","arguments":{"instruction":"classify"}}', self.tools)
        self.assertEqual(action["instruction"], "classify")

    def test_six_tools_graph_and_message_ids(self):
        actions = [
            {"tool": "analyze_image_quality", "arguments": {}},
            {"tool": "classify_image", "arguments": {"instruction": "Classify."}},
            {"tool": "localize_region", "arguments": {"instruction": "Locate."}},
            {"tool": "inspect_region", "arguments": {"bbox": [25, 10, 95, 70], "question": "Describe."}},
            {"tool": "segmentation", "arguments": {"bbox": [25, 10, 95, 70]}},
            {"tool": "answer_visual_question", "arguments": {"instruction": "Answer."}},
            {"answer": "finished"}]
        class Backend:
            scope = "fixture"
            def generate(self, prompt, *, state):
                return '{"answer":"finished"}' if state["router_phase"] == "graph_controller" else "fixture observation"
        graph = GraphAgentModel(Backend(), self.root / "trace.jsonl", max_tools=6, tool_runtime=self.tools,
                                coverage_plan=actions[:-1].copy())
        self.assertEqual(graph.generate("coverage fixture", state=self.state), "finished")
        episode = graph.episodes[-1]
        self.assertEqual(episode["tool_calls"], 6)
        self.assertEqual(episode["controller_calls"], 7)
        calls = [e for e in episode["events"] if e["node"] == "execute"]
        self.assertEqual(len({c["tool_call_id"] for c in calls}), 6)
        decisions = [e["decision_source"] for e in episode["events"] if e["node"] == "process"]
        self.assertEqual(decisions, ["prescribed_plan"] * 6 + ["model"])

    def test_coverage_rejects_extra_model_action_after_plan(self):
        class Backend:
            scope = "fixture"
            def generate(self, prompt, *, state):
                return '{"tool":"classify_image","arguments":{"instruction":"wrong next step"}}'
        graph = GraphAgentModel(Backend(), self.root / "trace.jsonl", max_tools=1, tool_runtime=self.tools,
                                coverage_plan=[{"tool":"analyze_image_quality","arguments":{}}])
        self.assertEqual(graph.generate("coverage", state=self.state), "")
        self.assertEqual(graph.episodes[-1]["status"], "failed")
        self.assertEqual(sum(e["node"] == "execute" for e in graph.episodes[-1]["events"]), 1)

    def test_model_selected_image_tool_without_coverage_plan(self):
        class Backend:
            scope = "fixture"
            calls = 0
            def generate(self, prompt, *, state):
                self.calls += 1
                return ('{"tool":"analyze_image_quality","arguments":{}}' if self.calls == 1
                        else '{"answer":"observed statistics"}')
        graph = GraphAgentModel(Backend(), self.root / "trace.jsonl", max_tools=2, tool_runtime=self.tools)
        self.assertEqual(graph.generate("inspect", state=self.state), "observed statistics")
        process = [e for e in graph.episodes[-1]["events"] if e["node"] == "process"]
        self.assertEqual([e["decision_source"] for e in process], ["model", "model"])


if __name__ == "__main__":
    unittest.main()
