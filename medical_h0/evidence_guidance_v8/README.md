# Evidence guidance v8

This small paired development diagnostic tests a fixed generic evidence-checking prompt on the mainline harness. Both arms receive all raw facts through the unchanged v7 input collector. R retains the v6 SYSTEM; I appends `adapter_v8.GUIDANCE`, covering target scope, requested-field relevance, consistent versus conflicting observations, all-operand citations, and accurate transformations. Tools, formulas, model/decoding, evaluator and per-episode budgets remain identical. There are no runtime checks against hidden expected evidence and no automatic parameter correction.

Six new evidence cases (three answerable, two conflicting, one missing) and six new mapping cases (four dose-times-frequency, one direct extraction, one explicit category) each run in R and I, for 24 episodes. Cases are interleaved with balanced arm order and fresh state. The v7 development results informed the policy, but no old panel is replayed or rescored. This prompt package is a combined intervention, not an isolated estimate for each checklist item and not HCL.

`PROTOCOL.json` freezes budgets, interpretation and stop rules. `test_design_v8.py` checks policy/pair isolation and all reference paths, including a numerically correct but incompletely cited derived dose that must remain untrusted. Every actual abstention and accepted calculation requires manual review after the run. Incorrect, unsupported and unattempted outcomes remain distinct.

Reuse v5-v7 and the recorded environment read-only, stage v8 in a fresh sibling run root, and provide the prior source paths via a private runtime config. Launch neutral copies of the worker/supervisor under the original first-attempt clock. This run stops after report/source delivery without starting HCL or another experiment.

Completed negative result: R 5/12 vs I 3/12; trusted mapping 1/6 in both arms. The guidance is not adopted. See [report](REPORT.zh-CN.md), [metrics](METRICS.json), and [manual audit](MANUAL_AUDIT.json).
