# Bounded executable H0 gate v5

One text-only, explicitly scoped H0 pass before any possible HCL work. The protocol is in `PROTOCOL_H0_V5.json`; data readiness and CPU checks are separate from model performance. Formula definitions reuse `../interface_diagnostic_v4` and its pinned calculator library. The solver reuses the recorded MedicalHarness LangGraph loop, HCL local Transformers loader, and Qwen-Agent Nous protocol.

Runtime modules are `run_v5`, `adapter_v5`, `tools_v5`, `runtime_v5`, `catalog_v5`, `qwen_transport`, and `budget_v5`. `audit_data`, `audit_rules`, and `score_v5` are evaluator-only. Linux `worker_v5` and `supervisor_v5` are launched through neutral entry filenames with private paths supplied by environment/config, under an independent time/storage/token guard. Source files are staged flat in a fresh run root beside the existing interface_diagnostic_v4 directory. Dependency roots, private model paths and GPU selection belong in a private runtime config. No patient fixtures, reference parameters, raw model traces, or deployment credentials are published.

The environment is an application-level allowlist on a trusted host, not an OS sandbox or the complete official tool environment. Parameters bind directly to visible facts; explicit result-to-category lineage is available for sPESI. General intermediate-result dependency graphs and visual tasks are outside this gate.

Data were previously model-evaluated. The 13 disjoint source-task groups do not establish 13 independent patients or fresh confirmation. This gate cannot certify clinical safety, full benchmark coverage, HCL benefit or a fair comparison against the earlier six-target diagnostic. Even a pass stops at this report; no HCL is launched.
