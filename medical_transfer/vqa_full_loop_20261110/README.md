# Medical VQA: complete HCL feedback and gated updates

The real HCL loop completed with one accepted memory update and explicit rejections/rollback. On 200 previously unused source groups, the selected learned state answered 74 correctly versus 57 for the no-memory harness (+8.5 percentage points). It did **not** beat the fixed six-example control, which answered 77 correctly. The purpose of this run was an honest interview demonstration, not a claim of clinical utility or a publication-ready method.

## Final confirmation

| Arm | Correct / 200 | Exact-match rate |
|---|---:|---:|
| B: no-memory static harness | 57 | 28.5% |
| F: fixed six-example static harness | 77 | 38.5% |
| H: restored historical state | 66 | 33.0% |
| N: selected newly learned state | 74 | 37.0% |

| Paired N comparison | Gain | Loss | Unchanged | Net pp | Paired bootstrap 95% interval, pp |
|---|---:|---:|---:|---:|---:|
| N vs B | 18 | 1 | 181 | +8.5 | [4.5, 12.5] |
| N vs F (preselected primary control) | 27 | 30 | 143 | -1.5 | [-9.0, 6.0] |
| N vs H | 25 | 17 | 158 | +4.0 | [-2.0, 10.5] |

All four arms completed all 200 cases with valid answer formats. Exact match is the original task metric; case-insensitive whitespace-normalized text, integer counts and order-invariant semicolon sets are handled by the existing evaluator. No synonym or answer-based repair was added. An exact-match advantage is not evidence of improved clinical reasoning or visual understanding. The numerical +8.5 pp happens to match an earlier experiment, but these are a different, newly reserved 200 groups, a new shared answer interface, and different absolute scores; do not merge the two results.

## What actually learned

Two predetermined starts, fresh and resumed, each received two batches of 24 new feedback examples. The original HCL training loop performed:

1. Reference evaluation on the current gate and historical anchors.
2. Real model answers, labeled feedback, raw-memory updates, abstract consolidation and the original skill-evolution call.
3. Memory-candidate evaluation, acceptance or restoration.
4. Model generation of two memory-selector and two router-context candidates per batch, structural checks, gate evaluation, acceptance or rollback.
5. State checkpointing and read-only development evaluation.

| Start / batch | Current correct before → candidate, /48 | Current gain / loss | Historical gain / loss, 72 anchors | Memory decision |
|---|---:|---:|---:|---|
| Fresh / 1 | 8 → 13 | 5 / 0 | 2 / 0 | Accepted |
| Fresh / 2 | 13 → 9 | 0 / 4 | 0 / 11 | Rejected and restored |
| Resumed / 1 | 11 → 10 | 3 / 4 | 0 / 11 | Rejected and restored |
| Resumed / 2 | 11 → 13 | 6 / 4 | 0 / 11 | Rejected and restored |

All 16 generated component candidates (eight per start) were rejected. The deployed change is therefore **raw/abstract memory**, not newly accepted selector/context templates, a learned graph topology, new model weights or new skills. Final fresh memory contains 10 raw records, two abstracts, 72 evaluator-only anchors and zero skills. Its artifact version remains zero; memory changes are recorded separately. The resumed arm retained its original memory and artifact version one. Stored state comparisons verified that the second batch restored the first-batch memory exactly for both arms.

This is a successful execution/update demonstration and a positive comparison with no memory. It is not evidence that the more elaborate mechanism outperforms a fixed few-shot prompt.

## Frozen data and selection

- 48 feedback-learning groups, 48 gate groups, 80 development groups and 200 confirmation groups; one example per group across the 376 selected groups.
- Existing identities excluded 1,674 historically exposed groups/pixels. Selected group/pixel identities are disjoint across study roles and from the historical exclusions. This does not establish patient independence.
- Original validation had insufficient unused question-type coverage. The new confirmation panel was reserved from unused `train_replay` sources, with the study-only role change recorded before model execution and member IDs fixed before target joins. It is **not the original test set**. Original test and sealed MedRAX panels were not used.
- Confirmation quota: 60 presence questions and 28 each for count, color, instrument, location and choice/other. Learning batches each contained four examples of each of the six types.
- Same frozen Qwen2.5-VL-7B-Instruct model, images, output contract and evaluator for all arms. The common short-answer contract requests digits for counts and semicolon lists for multiple items; it supplies no answer vocabulary. F receives six predetermined examples, the first training example of each type, without tuning.
- Development results: fresh 21/80, resumed 20/80, B 16/80, F 30/80, H 20/80. Before opening confirmation, the rule selected fresh as N and F as the stronger B/F comparator. The confirmation result did not influence this selection.
- Final confirmation was read-only; membership equality, all denominators, frozen parameters and unchanged learning states were checked. Bootstrap is 20,000 paired source-group resamples with fixed seed; intervals are descriptive, not an interview acceptance requirement.

## User-requested gate amendment

The user explicitly changed the goal from stringent research evidence to an interview demonstration while the first reference pass was still running, before any feedback-generated candidate result or acceptance decision existed. Version 2 retains a current gain of at least one correct answer out of 48, permits at most two historical old-correct losses out of 72 **per decision**, and requires at least 95% current format compliance. Invalid candidates, no-current-gain candidates and excessive historical losses still fail. Per-decision allowances must not be presented as a bound on cumulative forgetting.

The stricter attempt was archived, not erased. A completed B development pass was reused because its inference is independent of gate settings. Incomplete learning/F work was rerun and all calls retained in the original budget. The original clock was not restarted. The former +5 pp/positive-CI research criterion is retained only as a reference statistic and was not met against F. The single accepted memory update happened to satisfy the old stricter gate too.

## Cost and verification

- Wall time from the original start, including the amendment: 17,477.69 seconds (about 4 h 51 min), within the shared 23-hour hard timeout.
- 13,762 issued model calls across all arms/attempts; 13,759 completed call receipts. Three calls were interrupted by the explicit gate amendment and remain counted.
- Current-attempt calls completed without recorded generation failures. All seven final workers and the supervisor exited successfully; no owned process remains.
- Run-root storage at completion: 78,548,992 bytes, plus about 1 KB of neutral entry wrappers. Shared existing model/data/environment copies were read-only and are not new experiment storage.
- Peak allocated GPU memory was about 18.17 GB per worker. Confirmation calls: B/F 600 each, H/N 800 each. This run does not establish an efficiency advantage; training and rejected-candidate evaluation are substantial costs.

## Files and reproduction boundary

`results.json` contains only aggregate results, gate outcomes, state checks and cost receipts. `code/` contains the executed preparation, boundary, worker, supervisor, CPU-check and analysis logic, with deployment-specific executable paths replaced by `WORKER_PYTHON` and `WORKER_ENTRY` environment variables and the process-name denylist exposed as `FORBIDDEN_PROCESS_TERMS`. `protocol.template.json` replaces private resource locations with placeholders. Set `RUN_ROOT`, provide the authorized local resources, and use neutral worker entry wrappers before running. The supervisor refuses duplicate launches and expects a previously created original-clock `budget.json` and a passing `CPU_CHECK.json`.

The existing HCL package, medical evaluator/bridge and coordinate, semantic, task-scope and candidate-contract adapters are required. Relevant public modules live in sibling `coordinate_interface`, `semantic_boundary`, `task_scoped_memory`, `template_contract` and `frozen_vqa_confirmation` directories and `../src/medical_hcl`. Private image/question/answer panels, individual traces, learned memory, generated candidate text and historical deployment artifacts are intentionally not distributed. These omissions mean this directory is a source-and-aggregate evidence release, not a self-contained dataset/state replay package.

No further experiment is needed merely to claim the interview workflow ran. A future improvement must address the fixed-example control fairly and use a new confirmation panel; this already-observed panel must not become a tuning set while still being described as held out.
