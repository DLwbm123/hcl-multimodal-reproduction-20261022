# Component diagnostic v6

Completed: **COMPONENT_GATE_FAIL**. Submission passed 6/6; navigation passed 2/6; all six mapping cases were unattempted under the frozen gate. There were 19 real model generations. See [the Chinese report](REPORT.zh-CN.md), [metrics](METRICS.json), [closure audit](COMPLETION_AUDIT.json), and [synthetic-only tool traces](SYNTHETIC_EVIDENCE_AUDIT.json). No HCL or old benchmark replay was started.

Bounded synthetic diagnostics following the failed v5 H0. The v5 implementation and results remain frozen. This directory reuses its unchanged formula library, official inference transport, scorer and graph adapter; it repairs terminal-tool contracts, mutable-state error caching and evidence visibility in `runtime_v6.py`.

Run `test_components.py` in the recorded environment with the existing v5 dependencies. Its recovery regression reproduces the old cached-error failure, then requires the new implementation to succeed after reading the missing fact. The other checks cover terminal schemas, immutable/full-body success caching, durable evidence/cursors, source/output boundaries, and all navigation/mapping reference paths. No CPU check calls a model.

`fixtures.py` contains only explicit synthetic examples. The ordered real-model stages are submission, navigation and mapping, six cases each. Later stages run only after the previous stage passes all six cases. Seeded results and artificial fault states belong only to component testing. This is not another 18-target clinical benchmark and not an HCL experiment. See `PROTOCOL.json` for budgets and gates.

The runtime retains every fact actually read, not evaluator-selected evidence, plus page and observation indexes. There is no silent evidence truncation; the existing context cap remains enforced. This bounded design can raise context cost for large records; general long-record memory management is outside this diagnostic. Generation remains unconstrained Nous plus strict validation; no new grammar engine or serving stack is introduced.

`worker_v6.py` and `supervisor_v6.py` retain the previous independent resource guards and receive paths via a private runtime config. Stage sources flat in a new run directory beside executable_gate_v5, set its v5_code path, and use neutral worker entry filenames. Existing model weights and dependencies are read-only. Model-run source is committed before launch.
