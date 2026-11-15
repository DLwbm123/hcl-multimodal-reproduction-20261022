# Role-separated Agent execution completes, but the learning gain does not generalize

A frozen seven-arm, 200-case confirmation study found no improvement from either learning start. Fresh learning remained **36/200 before and after**; continuing from a previously learned state changed **74/200 to 73/200**. These are the two prespecified primary comparisons. Both Holm-adjusted exact McNemar p-values are 1.0. Engineering execution improved, but this experiment does not support a new learning benefit.

| Arm | Definition | Correct / 200 |
|---|---|---:|
| G0 | Empty initial Agent | 36 |
| GN | Same Agent after two feedback-learning batches | 36 |
| GT | Agent initialized with a previously learned direct-model state | 74 |
| GR | Same initialized Agent after two further learning batches | 73 |
| GF | Fixed six training examples, no adaptive learning | 60 |
| D0 | Direct model without memory | 56 |
| DT | Direct model using the same prior state as GT | 76 |

Each Agent arm completed 199/200 episodes. The failed episode remains in its 200-case denominator. GN-G0 had zero gains and zero losses; the paired bootstrap interval is [0, 0] percentage points. GR-GT had zero gains and one loss, delta -0.5 points, interval [-1.5, 0]. These intervals describe this observed panel, not a guarantee of zero population effects. Secondary comparisons are descriptive and unadjusted; GR exceeding GF or D0 does not establish improvement from this round, because GT already contains prior learning. The direct learned-state control DT scored 76/200, above GR's 73/200.

## What changed and what actually learned

LangGraph separates the action controller from an independent final-answer node. The controller sees the current image, question, tool schema and observations and returns a validated JSON action. The tool and final-answer paths receive the HCL guidance. A maximum of two tools and one action-format retry remains enforced; no fixed tool-coverage plan is prescribed. All six registered tools occurred in this confirmation panel. Model weights remain frozen.

Both learning starts ran two 24-case feedback batches with the original HCL memory consolidation, candidate generation, validation and historical-retention gates. Current validation requires at least one additional correct answer, at most two formerly correct historical anchors lost per candidate, and at least 95% output-format compliance. Cumulative forgetting is also reported.

Fresh initialization rejected both memory candidates: validation improved 6/48 to 10/48 and 14/48 respectively, but each lost three formerly correct historical anchors, exceeding the frozen allowance of two. All eight fresh prompt candidates were rejected. The committed state therefore remained empty raw/abstract memory, and GN and G0 have identical case-level success outcomes.

The initialized branch also rejected both memory candidates. The first lost three historical successes and reduced validation from 15/48 to 12/48; the second lost ten and reduced validation from 17/48 to 13/48. Two prompt-component updates were accepted during the first batch, increasing validation from 15/48 to 17/48; later candidates were rejected. Raw memory was unchanged. The final independent panel did not reproduce that validation gain: GR scored one case below GT.

Historical checks remained 26/72 for fresh and 35/72 for initialized learning, with zero cumulative losses or recoveries on those anchors. This must not be described as universal zero forgetting: independent confirmation still includes the one GR-versus-GT loss.

## Frozen protocol and runtime

New train48, selection48, development80 and confirmation200 roles were allocated from previously unused source/pixel groups, excluding 2,250 prior groups. All four roles are disjoint. The 200-case confirmation panel contains 60 yes/no-or-presence items and 28 of each of five other question types. It is a custom Kvasir-VQA panel, not an official test split; patient independence is unverified. Historical checks combine 24 classification, 24 localization and 24 VQA anchors; non-VQA history retains its prior direct interface. The seed is 20261115, intervals use 20,000 paired bootstrap draws, and both final committed states were locked before confirmation. No best-checkpoint, seed or outcome-based arm selection occurred.

All nine workers (two learning and seven confirmation) exited successfully. The full original-clock duration was 41,920.9 seconds (11.64 hours), including interruption and recovery. A temporary-file size-accounting race interrupted the first confirmation attempt; the narrow storage guard was repaired, evidence retained, and all confirmation arms rerun from the same frozen learning states. The deadline and learning results were not reset.

Total issued model calls were **30,137**, including archived attempts; **30,134** completed call receipts were recorded. The three interrupted in-flight calls remain counted. Final storage was approximately 177.6 MB, far below the 4 GB early-stop threshold. Per-arm costs, training-gate decisions and all comparisons are in `results.json`.

This negative result motivated no changes to the completed protocol. The separately started experience-applicability/tool-guidance study uses different data and must be reported separately. Previous 23/100 to 57/100 results belong to another panel and cannot be pooled or interpreted as a direct architecture comparison with this one.

## Release boundary

The release includes executed source, the temporary-file guard and recovery supervisor, tests, an additional aggregate-collection utility, a path-sanitized protocol template, and aggregate results. Deployment paths, medical images, private questions/answers, memberships, per-case traces, learned memory and candidate text are excluded. The template retains the original descriptive `read_after` typo saying six arms; the seven-arm list and executed seven-arm evidence are authoritative. Runtime still requires the existing HCL and medical-adapter packages, model and authorized data; this is not a self-contained reproduction bundle. Do not tune against this completed confirmation panel.
