# Input acquisition ablation v7

Twelve new synthetic navigation cases are paired across A (model-controlled reading) and B (all raw facts read by the program). Both arms use the unchanged v6 SYSTEM, tool schemas, dispatcher, task prompt, model, decoding and scoring. Only initial input acquisition differs. B retains distractors and conflicting records; no reference selection or parameter values enter its context. All reads and their costs are recorded separately. Each episode uses fresh state.

Six previously unattempted v6 mapping fixtures run independently under B; navigation scores do not gate them. This is a new diagnostic protocol and does not update v6. The 30-episode schedule alternates A/B order by case and interleaves one mapping probe after every two navigation pairs. Global engineering/budget failures can stop remaining episodes; task mistakes cannot.

`PROTOCOL.json` defines the frozen limits, decision rules, semantic audit and stop conditions. `fixtures_v7.py` defines all synthetic inputs and evaluator rules. `test_design_v7.py` checks pairing, raw-input isolation, scope/false-termination scoring and reference paths without a model. `evaluate_v7.py` produces preliminary structural scores; abstention reasons and every accepted computation require post-run semantic review. No evaluator feedback is returned to the model.

Reuse v5/v6 and their recorded dependencies read-only. Stage these files flat in a fresh sibling run root, supply v5/v6 paths in the private runtime config, and launch copied worker/supervisor files through neutral entry names. Preserve the first-attempt budget clock. This code has no HCL entry or automatic follow-on experiment.
