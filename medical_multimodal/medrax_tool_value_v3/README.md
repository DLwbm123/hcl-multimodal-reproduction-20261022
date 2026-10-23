# Fixed tool-evidence diagnostic

V2's consultation instruction raised successful tool-use episodes from 1/12 to 11/12 but reduced correct answers from 6/12 to 4/12. The prompt branch is closed. This independent development diagnostic asks whether the available tools provide useful question-relevant evidence at all, before investing in HCL updates.

Twelve fresh case groups receive three conditions: native agent, one fixed classifier call, or one fixed CheXagent call conditioned on the question. The two seeded conditions use the original MedRAX `execute_tools` method and then its unchanged graph. Seed calls count toward budgets and are reported separately from subsequent autonomous calls. All conditions receive identical prompts and canonical RGB PNG files. This format adaptation addresses an observed single-frame GIF stored with a JPG suffix; originals and prior failures remain intact.

The frozen protocol reports all 36 episodes, paired gains and harmful changes, runtime failures, actual vision payloads, planner cost and expert generations. At least two additional correct answers for the expert condition, ten successful expert seeds, and complete execution are a descriptive headroom criterion. Forced calls cannot pass autonomous-agent qualification, and no result establishes HCL benefit. All 36 development case groups across v1–v3 are excluded from future validation/test.

No new model downloads. Reuse v1 assets and environment read-only; aggregate route storage remains below 30 GB with a 28 GB early stop and 1 GiB reserve. One independent hour includes preparation and any retries, bounded by the existing route deadline. The single 30-minute monitor continues analysis and delivery.

Before inference run `prepare_cpu.py` and `check_design.py` using `RUN_ROOT`; launch through `control.py` with neutral worker entries. The latter fixture checks the real native graph, tool/result correlation, identical paired inputs, forced-versus-autonomous accounting, and an injected failure in the full denominator. Its synthetic scores are not experiment results.
