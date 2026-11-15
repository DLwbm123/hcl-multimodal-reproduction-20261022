# Learned experience helps, but applicability features and tool statistics add no confirmed accuracy gain

The prespecified primary comparison on a new 100-case panel was **41/100 for the new method versus 41/100 for the matched previous method**. There were two paired gains and two losses, delta 0 percentage points, bootstrap 95% interval [-4, 4], exact two-sided McNemar p=1.0. The result does not establish an incremental benefit from the new modules.

| Arm | Definition | Correct /100 | Queries with reference | Queries with tool statistics |
|---|---|---:|---:|---:|
| G0 | Initial empty-memory Agent | 15 | 0 | 0 |
| GR | Previous exact-question method, learned on the same two batches | 41 | 69 | 0 |
| GN | New applicability retrieval and training-tool statistics | 41 | 89 | 70 |
| GP | GN final state, controller statistics disabled | 41 | 89 | 0 |
| GE | GN final state, retrieval restored to exact matching | 41 | 69 | 70 |
| GF | Fixed six training examples | 42 | 0 | 0 |

All six arms completed 100/100 Agent episodes, with no invalid-action retries. Each used exactly two tools per case. Controller statistics changed the chosen tool mix (for example, whole-image VQA calls increased from 143 in GP to 164 in GN), but did not reduce tool use or produce a net accuracy benefit. Increased reference coverage did not yield a net accuracy benefit either. Equal total scores do not mean identical predictions: GN-GP had one gain and one loss, while GN-GE had two gains and two losses.

GN-G0 is a secondary positive contrast: +26 percentage points, 27 gains and one loss, interval [17, 35], nominal p=2.1607e-7. It supports the learned overall state relative to empty experience on this panel, not the novelty or utility of the new modules. GN-GF is -1 point, interval [-8, 5]. Secondary comparisons are descriptive and not multiplicity-adjusted. The earlier 23/100 to 57/100 study used different cases; do not pool the panels or interpret their absolute scores as a matched comparison.

## Method and accepted updates

Both methods use the same frozen Qwen2.5-VL-7B, LangGraph controller/execution/final-answer loop, two 24-case training batches, 48-case acceptance set, 72 historical anchors, raw-memory capacity 24, and two-tool/one-retry maximum. Both start with empty raw/abstract memory and generate one candidate per prompt component per batch. The old method is a matched former mainline method, not a separately tuned baseline.

The new retrieval filters by visible question intent and object family, with exact-question matches always allowed. Non-exact matches require the same known intent/object set and a frozen-text similarity above a threshold calibrated from the first 24 visible training questions (different-intent pairs, 95th percentile). Eligible records rank by 0.7 text cosine plus 0.3 frozen-image-feature cosine. At most two corrected training examples are supplied. These are frozen-model features, not a separately trained embedding model. Applicability calibration uses no training answer or validation outcome.

Training trajectories add intent, tool sequence, first-observation correctness and final correctness to raw-memory candidates. The fields share the memory commit/rollback boundary. The controller receives aggregate tool patterns with at least two supporting training episodes, never historical answer text. Retrieved reference images remain on the final-answer path. GP and GE are same-state inference ablations, not independently retrained methods.

Both methods accepted both memory candidates. New-method acceptance accuracy progressed 7/48 to 15/48 to 17/48; the previous method progressed 7/48 to 13/48 to 17/48. The new method also accepted one router-context update in batch two. The other new prompt candidates and all previous-method prompt candidates were rejected. Historical checks were 26/72 to 38/72 for GN (one loss, thirteen recoveries) and 26/72 to 37/72 for GR (one loss, twelve recoveries). The original per-candidate historical-loss allowance of two and current-format threshold of 95% were unchanged. Neither result establishes zero forgetting.

## Protocol, costs and limitations

Train48, acceptance48, development24 and confirmation100 were newly allocated before inference, excluding 2,970 prior/concurrent source groups. All 220 source/pixel groups are distinct. The confirmation panel contains 30 yes/no-or-presence items and 14 of each of five other question types. It is a custom Kvasir-VQA split, not the official test; patient independence remains unverified. Final committed states were locked before any confirmation evaluation. Seed20261115 and 20,000 paired bootstrap draws were fixed; there was no best-checkpoint or confirmation-driven configuration selection. All 600 confirmation outcomes, memberships, training-only reference provenance and read-only/frozen-state checks passed.

The run finished in 13,753.2 seconds (3.82 hours). All eight workers exited successfully. Total generation calls were 18,917 issued and completed, plus 473 frozen-feature forward calls taking 43.2 seconds. Final storage was approximately 138.3 MB. Generation time summed across workers is not wall-clock duration. GPU sharing and load differ across roles, so timing differences do not establish controlled speedup. Per-role generation and feature costs are retained in `results.json`.

Development accuracy was 9/24 for each method; location questions were 0/4 and color questions 1/4 in each. The next evidence-contract experiment was designed and locally frozen from those development errors and prior project contract diagnostics before inspecting these confirmation metrics. The separate action-value experiment also uses training pairs, new acceptance cases and separately reserved confirmation cases. Neither changes this completed study.

This release contains executed source, checks, a path-sanitized protocol, allowlisted aggregate results and the aggregate collection utility. The runtime still depends on the existing HCL and medical-adapter packages, frozen model and authorized data. Medical images, private questions/answers, case-level traces, memberships, learned state, candidate text and deployment paths are excluded. Do not tune against this completed confirmation panel.
