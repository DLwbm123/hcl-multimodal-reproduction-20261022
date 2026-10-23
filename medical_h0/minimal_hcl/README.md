# Restricted workflow-only HCL pilot

See REPORT.zh-CN.md and protocol.json. Run was interrupted by a storage accounting race; no final efficacy result. Exact exercised execution/optimizer adapter and synthetic fixtures are included. Historical H0 failure remains unchanged.

Reuses the Qwen-Agent and MH-Lab dependencies pinned in ../official_tool_call. hcl_core is copied from this repository’s HCL optimizer, prompt template and JSON modules. Candidate transport is adapted to the official Nous function-call schema; only workflow prose is learnable.

CPU tests: `python test_learning.py /path/to/medicalharness` and `python -m unittest test_tools test_transport`. Production launchers with private host paths and raw trajectories are excluded.
