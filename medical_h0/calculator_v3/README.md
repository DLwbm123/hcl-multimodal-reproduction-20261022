# Calculator-backed MedicalHarness H0

See START.zh-CN.md and protocol.json for the frozen acceptance and conditional HCL protocol. This is a research adaptation, not a full official shell-environment reproduction or clinically certified service.

`accept_and_run.main(run_root, medicalharness_repo, hcl_repo, model_path)` runs the fixed learning-only capability gate and, only if it passes, the Static/HCL workflow trial. `run_root/data/visible_split.json` and evaluator-only `evaluator_gold.json` are prepared privately from the previously frozen 32-task split. No benchmark records are included in this release. Use the original run protocol for paths, independent supervision and resource limits; deployment-specific entrypoints are kept private.

Dependencies reuse the preceding `../runtime_v2` runtime: Qwen-Agent Nous, Transformers, LangChain/LangGraph, jsonschema, the pinned MedicalHarness and HCL optimizer. No new inference dependencies are installed. The bundled calculator domain library is Apache-2.0, pinned by `vendor/medical_calc/PROVENANCE.json`; only domain/shared Python code is imported. Local formula changes are in `LOCAL_PATCHES.json`. Its test fixtures are upstream synthetic unit examples, not MedTool cases.

`formula_coverage.json` records 33 upstream pure-domain test files and the 85 calculator implementations actually executed. The 152-entry directory exposes coverage status; only tested, typed entries execute. 1024 upstream tests passed. `test_clinical_contract.py` independently checks formula boundaries, identity, numeric scoring and recovery after errors. Run the source tests with pytest and the adapter tests with unittest; `test_learning.py` additionally takes the MedicalHarness repo path. `verify_upstream.py` accepts a JSON list of selected upstream test paths and a receipt output path.

The evaluator uses explicit aliases and episode identities instead of fuzzy token overlap. Any analysis comparing older protocols must retain their original score and separately label the new score. The reused final panel cannot establish an untouched independent confirmation.
