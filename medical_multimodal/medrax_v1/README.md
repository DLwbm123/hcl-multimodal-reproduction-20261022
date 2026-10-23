# MedRAX native development qualification

This route uses the upstream MedRAX LangGraph agent and four actual chest X-ray
tools: DenseNet classification, PSPNet anatomical segmentation, CheXagent VQA,
and the two-head vision encoder–BERT report generator. Upstream source is pinned at
`dae30e2f136ef0b2a40885a4c335386e9ffad052` in
[MedRAX](https://github.com/bowang-lab/MedRAX). Dataset metadata/images come from the
public [ChestAgentBench dataset](https://huggingface.co/datasets/wanglab/chest-agent-bench),
with the revision and exact file sizes in `asset_plan.json`.

The planner is an existing, read-only local Qwen2.5-VL-7B-Instruct checkpoint.
This is a local-planner adaptation, not a reproduction of the published GPT-4o
MedRAX score. Its visual chat template is retained; the transport adds function
definitions and serialized tool calls because the local checkpoint's template
does not implement the `tools` argument. Actual images pass through the vision
processor. Malformed calls fail without JSON repair. No paid API is used.

## Frozen scope

`native_protocol.json` fixes a 12-question development panel before any model
generation. It includes one image and one question per case, balances category
exposure using metadata only, and excludes all selected cases from future
validation/test use. This is a subset of the 2,500-question catalog, not a full
benchmark evaluation. Gold answers and explanations never enter planner or tool
messages. All 12 items remain in the denominator, including not-run cases.

The first image is also used for four separately recorded direct tool checks.
These are engineering checks, not agent successes. The agent panel then runs
through the unmodified upstream graph. Report model IDs are resolved to pinned
local assets; xrayvision's cache is redirected inside the run root. Tool schemas,
descriptions, predictions, and results remain native. A boundary wrapper limits
tool inputs to images supplied in the current episode and records failures.

Qualification requires all four direct tools to execute, actual vision input on
every planner call, at least 3/12 correct answers, at least four valid final
answers, and at least three episodes with a successful real tool call. These
thresholds establish development feasibility only. They do not establish HCL
efficacy or clinical validity. Adaptive HCL is gated on this qualification.
The disabled HCL adapter is a literal passthrough and has a CPU identity check.

## Implementation and execution

- `prepare.py`: resumable pinned downloads, image extraction, isolated environment.
- `supervise.py`: independent original-clock and allocated-storage guard.
- `vision_transport.py`: image-aware Qwen/LangChain conversion and strict parsing.
- `native_runtime.py`: upstream imports, local asset resolution, bounded tool logging.
- `run_native.py`: direct tool checks, native graph evaluation, complete-denominator scoring.
- `check_transport.py`: CPU vision payload, call/result correlation, and disabled-update checks.
- `continue_native.py`: waits for preparation, checks dependencies/GPU memory, then runs qualification.

All runtime paths are supplied through `RUN_ROOT` and an untracked
`runtime.private.json`. The latter identifies existing shared environments,
planner weights, and the selected GPU. Neutral entry scripts outside this source
tree configure environment variables and start workers; visible process arguments
contain no project or user identifiers. Existing shared models/environments are
read-only. A single external 30-minute monitor follows preparation, qualification,
analysis, and any subsequently frozen HCL experiment.

Preparation and retries share the initial 23-hour early deadline. Native execution
also has a one-hour hard limit. New allocated storage includes downloads, caches,
environment, images, temporary output, and logs, with the user-updated 28 GB early stop including
a 1 GiB reserve and a strict exclusive 30 GB ceiling, aggregated over this route
and future registered run roots. `RESOURCE_POLICY.json` overrides the original
asset planning figures. An independent live watchdog applies the tighter cap
without restarting existing workers. Initial download volume is 14.21 GB; the
nested XraySigLIP dependency discovered during constructor loading adds 2.613 GB. No MIMIC dataset download,
MAIRA, LLaVA-Med, or RoentGen weights are part of this run.

## Current evidence

CPU checks passed on the existing environment with zero model generations:
nonempty vision tensor `[1024, 1176]`, strict malformed-call rejection, matching
tool-call/result IDs, and unchanged input/output identity with HCL disabled.
The fourth attempt completed all 12 native tasks: 4 correct, 12 valid final
answers, and only one episode with a successful real tool call. All four direct
tools passed. The frozen native qualification gate therefore failed. See
[the complete report](REPORT.zh-CN.md) and `native_report.json`. HCL updates
remain disabled. A fresh-case paired consultation diagnostic is separately
frozen in `../medrax_tool_consult_v2`.

Raw questions, images, model assets, private runtime configuration, full model
outputs, and patient-derived content stay outside this public source tree.
Only permitted source, protocol, aggregate reports, and reproduction instructions
will be published. Upstream code and asset licenses remain with their owners.
