# Native MedRAX with a local Qwen controller: development diagnostic

The complete 96-cell attempt matrix produced **no positive method result**. See `REPORT.zh-CN.md`, `AGGREGATE_REPORT.json`, and `DELIVERY_CHECK.json`. Ten model-format failures remain failures.

`code/worker.py` uses the upstream MedRAX Agent via `native_runtime.py`, four existing chest-X-ray tools, the shared image/token/cache guard, and `local_controller.py`. `run_agent.py` is the existing runtime module reused by the worker; its old subscription and six-tool entry point are not used in this experiment. LocalController replaces SubscriptionModel before execution. `policy.py` is unchanged historical H0/Raw/Fixed/Proposed_v0 policy selection. This is not an HCL learning run or parameter RL.

Runtime used the existing Transformers 4.51.3 native-tool environment. Supply existing licensed local assets under RUN_ROOT: `upstream/medrax`, `models/{findings,impression,chexagent,xraysiglip}`, and `cache/torchxrayvision`; private approved inputs/raw-bank/candidate/targets under `private`. Targets are opened only in terminal scoring. The unshared input manifest contains id, source_key, question, images, image_labels, privacy_approved, and reference_tier. Do not infer dataset access from the public code.

Set R981_ROOT to RUN_ROOT, R1136_TMP and R1136_CACHE inside it, LOCAL_MODEL_PATH to the local Qwen checkpoint, CUDA_VISIBLE_DEVICES to the checked GPU, and HF offline/module-cache variables. Provide `budget.json` with started_epoch, hard_deadline_epoch, stop_new_bytes. Run through `supervisor.py` with NEUTRAL_PYTHON/NEUTRAL_WORKER pointing at deployment entry points. The original clock must survive resumption. Use the frozen `code/protocol.json`; each arm sees the same budget and tools.

The public copy replaces two machine-specific legacy paths in `run_agent.py` with NATIVE_PYTHON_INCLUDE_PATH (optional build headers) and LOCAL_CLIP_MODEL_PATH (only the unused six-tool loader). No method logic was changed for publication. Deployment-specific launch files and all private data are excluded.

Minimal software check: `python code/local_controller.py`; this verifies accepted actions, rejection of unknown tools/fields/types, and exact final-choice handling. Actual image/model/tool operation was checked on the private authorized development data; a software self-check alone is not a clinical validation.
