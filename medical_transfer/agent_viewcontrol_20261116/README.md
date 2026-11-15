# Training controls found no crop accuracy gain

Two unchanged incumbent reruns on the same GPU returned **25/48** each, with identical answer strings and zero gains/losses. The matched ROI-crop and whole-image inspection arms both returned **4/16**, with identical per-case correctness and zero gains/losses. Four answer strings changed, without changing correctness. All 16 region proposals agreed between view arms: five legal boxes, ten absent and one invalid; each arm executed five successful inspections and eleven whole-image fallbacks.

On these 16 training cases the incumbent scored 3/16; each macro scored 4/16. This is reused training evidence, not independent validation, and does not support crop accuracy benefit. It does not establish that crop and whole-image observation are equivalent in general. A/A agreement on these 48 cases does not resolve the previous acceptance discrepancy or prove universal determinism.

## Controlled execution

Prespecified sequence on GPU4: incumbent repeat A48, incumbent repeat B48, crop16, fullview16. Same seed2026111607, fresh process per arm; all cases come from existing training48, in unchanged order. For legal proposals, fullview changes the actual inspection rectangle to the complete original image; proposal content, tool prompts, finalizer, frozen model, HCL state and contracts are otherwise unchanged. Geometry metadata truthfully changes too, so this is not a pixels-only intervention. Absent/invalid proposals keep the original fallback. All macro episodes use two observations and zero model-controller calls. No policy fitting, selection, deployment, historical evaluation or confirmation occurs.

The source run consumed the last selection24 and confirmation76. Neither was copied or evaluated here. Official test remained unused; patient independence remains unverified.

## Verification, failure recovery and costs

All four workers exited zero and passed real-model preflight; original-pixel inspection receipts, identical proposals, fixed denominators, training-only references, frozen weights and read-only state were verified. Each unchanged training arm has one failed Agent episode retained in its 48-case denominator.

The automatic summarizer initially failed because failed Agent episodes lack the top-level controller_calls field. The source fix counts actual process events, including failed attempts. Original supervisor failure and error evidence remain intact; offline recovery generated the complete analysis with **zero new model calls**, within the original two-hour clock. analysis_recovery.json records this distinction; successful workers are not misrepresented as an initially successful supervisor.

Inference plus initial analysis attempt took 1,262.33 seconds (21.04 minutes); recovery completed 1,521.75 seconds after the original start. Recorded inference storage was 9,400,320 bytes. Including all preflights: 1,002 successful generations, 222 feature forwards (13.06 seconds), 268 tools, 200 controller calls. Shared-GPU timing is descriptive, not a controlled speed benchmark.

Published source, sanitized protocol, aggregate results and recovery receipt exclude private images/crops, questions/labels, membership, per-case traces/boxes and learned state.
