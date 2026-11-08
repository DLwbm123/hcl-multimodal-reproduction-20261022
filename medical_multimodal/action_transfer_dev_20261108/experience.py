"""Compile a development-only operation from frozen feedback; no patient data.

Predicate/fields/candidate prompts are specified by the protocol. The data
select and qualify an operation; they do not discover a new clinical rule.
"""
import json
from pathlib import Path


def compile_candidate(protocol, aggregate):
    if not aggregate["complete"] or not aggregate["candidate_gate_pass"]:
        return None, "rejected_by_frozen_gate"
    metrics = aggregate["metrics"]
    best = max(metrics, key=lambda arm: metrics[arm]["correct"])
    if best != "focused" or metrics[best]["correct"] <= metrics["report_with_decision"]["correct"]:
        return None, "no_supported_focused_advantage"
    return {
        "schema_version": 1,
        "status": "qualified_for_agent_development_only",
        "predicate": "airspace_opacity_or_consolidation_presence",
        "modality": "chest_xray",
        "image_count": 1,
        "output_type": "boolean_observation",
        "action": {"tool": "chest_xray_expert", "prompt": protocol["arms"][best],
                   "max_new_tokens": protocol["max_new_tokens"]},
        "evidence": {"source_groups": protocol["source_groups"], "views": protocol["image_views"],
                     "correct": metrics[best]["correct"], "paired": aggregate["paired"],
                     "reference_type": "source_report_concordance_proxy"},
        "clinical_correctness_established": False,
        "learning_advantage_over_strong_fixed_established": False,
        "manual_parts": ["predicate", "applicability fields", "candidate prompt space", "gate"],
    }, "qualified_by_new_development_gate"


def select_action(entry, request):
    """Request uses only current visible task information, never its answer.

    The caller must separately verify that the declared predicate matches the
    task; this function does not infer intent or establish clinical truth.
    """
    if entry is None:
        return None, "no_qualified_entry"
    for field in ["predicate", "modality", "output_type"]:
        if request.get(field) != entry[field]:
            return None, "outside_supported_" + field
    images = request.get("image_paths", [])
    if len(images) != 1:
        return None, "requires_explicit_multiview_plan"
    if request.get("privacy_approved") is not True:
        return None, "image_review_required"
    return {**entry["action"], "image_paths": list(images)}, "selected_development_operation"


def write_candidate(destination, entry):
    """An unqualified candidate cannot overwrite the current state."""
    if entry is None:
        return False
    destination = Path(destination)
    tmp = destination.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(destination)
    return True
