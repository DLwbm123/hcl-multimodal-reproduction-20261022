# Observation provenance and admission diagnostic

See REPORT.zh-CN.md for the post-hoc audit, SUMMARY.json for aggregate counts, INTERVIEW.zh-CN.md for evidence-aligned interview notes, and protocol.json for the frozen source-only experiment.

The worker reuses base_worker.py, native_runtime.py, local_controller.py, qualified_tool.py, cache_compat.py, token_budget.py, policy.py, run_agent.py and supervisor.py from ../report_feedback_20261110/code without changing the model/tool configuration. Stage those support modules with audit.py, admission.py, worker.py and protocol.json in a run-local code directory. The same existing neutral launcher/supervisor pattern sets the run root and local model assets. Required private inputs are inputs.json, qualified_targets.json and cached_reports.json. Never put these inputs in a public repository.

Run audit.py --self-check and admission.py for the small synthetic checks. Audit claims are not clinical truth. The admission experiment is a deterministic component diagnostic; it does not train, promote an HCL candidate or establish novelty. The source46 is reused, not new independent data. No weight downloads or new tool search.
