# Native MedRAX paired action feedback acquisition

This source-learning run pairs the existing Fixed workflow with an extra per-image CheXagent observation before the same native Agent. It collects action feedback for subsequent HCL memory and routing. It does not itself train or accept a learned policy, update parameters, or evaluate an independent test set.

The frozen denominator is 46 source groups, 81 approved image files and two arms (92 episodes). Only 13 source-aligned visual-discriminator groups provide benchmark agreement feedback; 33 groups have unknown semantic feedback. The reference qualification is not independent clinical adjudication. Etiology or management language in an option is not established by visual agreement.

`code/worker.py` executes and scores the paired run. `code/base_worker.py`, `run_agent.py`, `native_runtime.py`, the controller and guards reuse the previous native runtime. The old subscription/six-tool entry points are unused. The current arms share native textual-answer handling, an eight-call total limit, guarded image paths, token/visual contracts and strict final choices. These are interface repairs, not method contributions or retrospective repairs of previous scores.

Provide authorized private inputs with id, source_key, question, images, image_labels, privacy_approved, reference_tier, semantic_feedback_allowed, tool_eligible_images and composite_image_files. Only qualified source-learning targets go in private/qualified_targets.json; targets are opened in terminal scoring. Do not provide development/test labels. Private observations, raw traces and original image data are deliberately unshared.

Use the asset layout and environment from the previous local-controller run: existing upstream/medrax; models/{findings,impression,chexagent,xraysiglip}; shared cache/torchxrayvision; independent run cache/module_cache/tmp/outputs. Set R981_ROOT, R1136_TMP, R1136_CACHE, LOCAL_MODEL_PATH, CUDA_VISIBLE_DEVICES, offline Hugging Face variables, NEUTRAL_PYTHON and NEUTRAL_WORKER. Use supervisor.py with budget.json (started_epoch, hard_deadline_epoch, stop_new_bytes), preserving the original clock across any repair. Frozen protocol: four hours total, per-case 900 seconds including probes, eight tool calls, 10GB apparent directory stop; actual new storage must remain below 50GB. Deployment wrappers are private.

The public runtime replaces two legacy host-specific paths with NATIVE_PYTHON_INCLUDE_PATH and LOCAL_CLIP_MODEL_PATH, as in the prior public run. No experiment method logic is changed by that substitution.

Software checks: `python code/local_controller.py` and `python code/check_scoring.py`. Actual model/tool operation is supported only by the aggregate result and private receipts, not by these checks alone. Image-similarity audit was triggered by observed cross-ID duplicates; its 64px correlation threshold cannot rule out all transformed duplicates or establish patient independence.
