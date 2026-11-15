# Independent tool observations with output contracts failed acceptance

The sole prespecified candidate was rejected: **20/48 to 16/48** on new acceptance cases. Historical correctness changed from **38/72 to 37/72**, with two correct-to-wrong changes and one recovery. Current format compliance was 100%. The required net gain of at least one correct answer was not met. No independent confirmation or shifted-image inference ran, and no candidate was deployed.

| Arm | New acceptance /48 | Historical /72 |
|---|---:|---:|
| Frozen learned predecessor | 20 | 38 |
| Training output-contract ablation | 21 | 38 |
| Contract plus independent tool observations (sole candidate) | 16 | 37 |

The contract-only ablation was explicitly ineligible as a fallback. Its one-case gain does not establish a validated improvement. No gate was relaxed and no arm was selected after seeing these scores.

The candidate had six gains and ten losses relative to its predecessor. The eight-case question-type strata were heterogeneous: count improved 3 to 6, color 0 to 1, instrument 3 to 2 (two gains and three losses), location 1 to 0, choice/other 6 to 0, and yes/no-or-presence stayed 7. These are post-run acceptance diagnostics, not independent confirmation or a basis for claims about generalization. Any subsequent routing mechanism requires training-only fitting and new acceptance/confirmation cases.

## Mechanism and frozen protocol

This study reused the parent final learned memory/prompt state and 48 training feedback cases. It did not update model weights or call HCL.train. Training-only feedback supplied color and location output-label sets and completeness instructions, with eight examples supporting each type (four color labels and eleven location labels). Those lists describe permissible training-observed output styles, not evidence that a finding is present in a new image.

The contract-only ablation added those instructions to tool observations and finalization. The sole candidate additionally removed historical guidance and controller answer guesses from whole-image model-tool requests; crop-tool requests retained controller coordinates but used a neutral current-question observation request. Finalization still had learned HCL guidance and retrieved historical references. Controller action schema, two-tool/one-retry limits and memory state remained unchanged. All model nodes used the same frozen Qwen2.5-VL-7B.

The design was frozen from parent development errors before viewing parent confirmation results. New acceptance48 and reserved confirmation100 excluded 3,314 historical/concurrent source groups, with distinct source/pixel groups. Patient independence is unverified and the official test was not accessed. Gate thresholds were current gain >=1/48, historical losses <=2/72, and current format >=95%. Only the full candidate could qualify. It failed, so the 100 reserved confirmation cases were not evaluated.

## Verification, costs and limits

All three real-model preflights passed and all workers exited successfully. Each arm completed all 48 acceptance Agent episodes and 24 historical VQA episodes; the other 48 historical anchors retained their direct task interfaces. There were no failed Agent episodes. Each arm made two tool calls per VQA query. Each had historical references for 47/48 acceptance queries, and tool statistics for 31/48. Complete split denominators, read-only state, frozen weights, training-only reference provenance and absence of query self-reference passed the final aggregate checks.

Total costs including preflight: 2,208 generation calls issued and completed, 375 frozen-feature forwards (37.64 seconds), 444 tool calls and 222 completed Agent episodes. The study finished in 2,222.5 seconds (37.04 minutes), within its four-hour original clock; recorded storage was 17,174,528 bytes. Per-role wall time was affected by GPU sharing and does not establish speed differences.

A planned shifted-image panel never ran. Its design perturbs the entire Agent input: changing the current image can change controller actions and adaptive reference ranking. The reference source bank stays unchanged, but per-query references need not. It must not be presented as an isolated tool-vision or fixed-reference test.

This is controlled mechanism development, not evidence of a novel complete continual-learning algorithm. The output-contract idea already existed in this project. Uniformly removing guidance was harmful overall, despite gains in some question types. Results do not support deploying the candidate or claiming zero forgetting.

This release contains executed source, checks, path-sanitized protocol, aggregate results and the aggregate collector. Existing HCL/medical adapters, frozen model and authorized data remain runtime dependencies. Images, private questions/answers, memberships, individual trajectories, learned state and exact training-derived label lists are excluded. Inherited learning configuration fields are adapter settings, not new HCL training executed in this study.
