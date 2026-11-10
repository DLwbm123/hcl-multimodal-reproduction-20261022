"""Shared correctness guard for future arms; no learned state or clinical score.

Install after load_six() on a new run. Existing outer tool/cost guards remain.
The case state must contain only paths approved by the private image review.
"""
from pathlib import Path
from threading import RLock

from cache_compat import inspect_prefill
from token_budget import encode_checked


class QualifiedExpert:
    def __init__(self, tool, state, record):
        self.tool, self.state, self.record = tool, state, record
        self.name, self.description, self.args_schema = tool.name, tool.description, tool.args_schema
        self.lock = RLock()

    def invoke(self, args):
        evidence = {"admitted": False, "generation_attempted": False}
        try:
            parsed = self.args_schema.model_validate(args).model_dump()
            paths = parsed["image_paths"]
            allowed = {str(Path(p).resolve()) for p in self.state["images"]}
            if not paths or any(str(Path(p).resolve()) not in allowed for p in paths):
                raise ValueError("Image outside the approved current case")
            cap = parsed["max_new_tokens"]
            if not 1 <= cap <= 512:
                raise ValueError("Output token cap must be between 1 and 512")
            tokens, budget = encode_checked(self.tool.tokenizer, paths, parsed["prompt"],
                                            cap, self.tool.model.config.max_position_embeddings)
            evidence["token_budget"] = budget
            if tokens is None:
                raise ValueError("Request exceeds the configured context; explicit replanning required")
            evidence["admitted"] = True
            with self.lock, inspect_prefill(self.tool.model, normalize_empty_cache=True) as observation:
                evidence["generation_attempted"] = True
                result = self.tool.invoke(parsed)
            evidence["visual_observation"] = observation
            if observation["visual_calls"] != 1 or observation["visual_image_counts"] != [len(paths)]:
                raise RuntimeError("Visual encoding contract failed; observation withheld")
            evidence["visual_contract_pass"] = True
            return result
        except Exception as exc:
            evidence.update(error=str(exc), visual_contract_pass=False)
            return ({"error": str(exc)}, {"analysis_status": "failed", "clinical_evidence_eligible": False})
        finally:
            self.record("expert_qualification", evidence)


def install_shared_guard(tools, state, record):
    """Same installation for H0/Raw/Fixed/Proposed, before any new episode."""
    selected = [t for t in tools if t.name == "chest_xray_expert"]
    if len(selected) != 1 or not hasattr(selected[0], "tool"):
        raise ValueError("Expected one existing wrapped CheXagent tool")
    if isinstance(selected[0].tool, QualifiedExpert):
        raise ValueError("Shared guard already installed")
    selected[0].tool = QualifiedExpert(selected[0].tool, state, record)
    return tools
