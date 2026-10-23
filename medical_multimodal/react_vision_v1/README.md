# Native ReAct visual-evidence acquisition

The MedRAX evidence-adaptation route is closed after its fixed expert condition lost six previously correct answers. This independent route tests the published Lagent ReAct / AgentLego execution architecture on twelve new ChestAgentBench development case groups, before any HCL updates.

The text controller receives the clinical question and an image path. Original LLaVA image/region description and EasyOCR tools read the image pixels; the controller has no direct image tensor. This tool-mediated architecture is different from the previous directly visual MedRAX planner. The original ReAct English prompt, parser, ActionExecutor and loop are reused. The native tool implementations use the existing LLaVA v1.5 7B checkpoint, 336px preprocessing and 100-token generation limit. Calculator uses its original implementation behind an input boundary. GoogleSearch is unavailable and omitted equally for every episode. No per-case reference tool subset is exposed.

All 107 MedCTA cases remain reserved for external evaluation. They are not used for this experiment, adaptation, gate selection or prompting. This is a project-specific base-agent qualification on ChestAgentBench, not a MedCTA benchmark score or reproduction of its reported model results. It does not revive the closed synthetic-calculator experiments.

The frozen gate requires all four direct tool checks, at least 3/12 correct answers, 6/12 valid finals and 6/12 episodes with a successful real visual tool. Failures and unattempted cases remain in the denominator. Direct tool checks are separately costed and do not count as autonomous success. Only after qualification may a separately frozen, case-isolated Static/Ungated/HCL experiment proceed. All 48 development case groups across the two architectures are excluded from later validation/test.

The independent one-hour clock includes preparation and any setup repairs; the existing route deadline and aggregate 30GB ceiling also apply. Existing models and environments are reused read-only. A tiny isolated dependency overlay resolves missing imports without changing shared environments. Parent, controller and tool-worker command lines use neutral entries; the tool worker shares the controller process group for independent hard termination.

`prepare_cpu.py` canonicalizes the 12 single-frame images identically. `check_native.py` checks the actual ReAct loop with a real Calculator call and image/expression boundaries without model inference. `control.py` starts the guarded controller; `tool_worker.py` keeps the original visual tools in their compatible environment. Private raw questions, images, responses and runtime paths are excluded from public delivery.

Upstream sources: [MedCTA](https://github.com/IVUL-KAUST/MedCTA), [Lagent](https://github.com/InternLM/lagent), [AgentLego](https://github.com/InternLM/agentlego). Source revisions and case selection are in `protocol.json`.

Final status: **closed, native qualification failed**. All 12 episodes completed, 4 correct, 6 valid finals, zero autonomous visual-tool episodes. Four direct tool checks succeeded; those are separate engineering evidence. No HCL updates. See [Chinese report](REPORT.zh-CN.md) and [aggregate ledger](REPORT.json).
