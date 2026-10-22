# Fixed development memory ablation

Read [the Chinese report](REPORT.zh-CN.md), [aggregate results](RESULT.json), and [completion checks](completion_audit.json).

This is a read-only 36-case development diagnostic, not a new continual-learning run. Static and no-memory are one condition. All five conditions use the same frozen model; the four harness arms differ only in the available fixed memory pool. The model coordinate semantic gate remains failed.

The published worker is the executed worker with host paths replaced by runtime environment variables. Provide HCL_DIAGNOSTIC_ROOT and HCL_PREPARED_ROOT. Private input files are private_development.jsonl (36 validation examples), private_frozen_memory.json (the existing accepted raw10/abstract2), class_vocabulary.json, protocol.json and budget.json. The budget includes started_epoch, early_stop_seconds=6900, hard_limit_seconds=7200, max_calls=1600, early_stop_new_bytes=450000000, max_new_bytes=500000000 and a read-only model_path. Never reset the start time for a retry. No private inputs or model files are distributed here.

The supervisor also needs HCL_PYTHON and HCL_NEUTRAL_WORKER_ENTRY and reserves GPU4, matching the run. Deploy neutral wrappers according to your host policy; supply all runtime settings before dispatch. It must coexist with any stricter campaign-wide guard. Only the precise completed Direct prefix (36 rows / 36 calls) can be reused by the included engineering resume logic; arbitrary partial resumes fail closed. No production experiment launches during import or tests except explicitly running worker/supervisor scripts.

CPU helper checks: `python -B -m unittest discover -s medical_transfer/development_diagnostic -p 'test_*.py'`. Existing coordinate caller tests cover the shared boundary. Pipeline init order was additionally checked on the server for all four actual harness configurations without loading weights.
