# Scoped episodic Agent learning improves held-out accuracy; visual increment remains unproven

On a newly reserved 100-case confirmation panel, the full single-batch learning procedure improved normalized exact-match accuracy from **23/100 to 57/100**. The primary paired contrast was +34 percentage points, with 37 gains and 3 losses, bootstrap 95% interval [24, 45], and exact two-sided McNemar p=1.9465e-8. This is positive evidence on this panel; it is not an official-test or clinical-efficacy claim.

| Arm | Definition | Correct / 100 | Agent episodes completed / 100 |
|---|---|---:|---:|
| G0 | Initial empty-memory Agent | 23 | 99 |
| GN | Final learned state, corrected text and visual references | 57 | 99 |
| GX | Same final state and corrected text; reference images withheld | 56 | 99 |
| GF | Fixed six training examples, no learned memory | 47 | 99 |

All cases remain in the denominator, including each arm's failed episode. The primary GN-G0 contrast was declared before inference. Secondary GN-GX is +1 point (2 gains, 1 loss; interval [-2, 5]; p=1.0), so the extra images have no established incremental benefit. GN-GF is +10 points (20 gains, 10 losses; interval [0, 21]; nominal p=0.0987), which is not conclusive superiority over fixed examples. Secondary tests are not multiplicity-adjusted. Do not attribute the full gain to reference images.

## Learning mechanism and outcomes

The complete HCL feedback, candidate generation, current-validation gate, historical retention check, and commit/rollback loop is retained. Qwen2.5-VL-7B weights remain frozen. LangGraph explicitly separates tool decisions from final medical answers; the controller receives the current question, tool schema and observations, while retrieved guidance and optional historical reference images support the answer path.

New raw-memory retrieval requires the same normalized visible question. It removes incorrect historical attempts from the exposed retrieval view and uses corrected training feedback. Unknown questions receive no episodic reference. Up to two records are deterministically selected, preferring distinct corrected answers, and joined to an authorized training-only image registry. References are labelled as different images, with the current target explicitly placed last. Images appear only at the final-answer node. Historical non-VQA tasks retain their prior direct interface.

Memory capacity was increased from 10 to 24 raw entries per task. GX shares GN's exact final state, capacity and reference text, with only reference images withheld. G0 has no learned memory and GF has six examples; their memory/context budgets differ. The study does not isolate all individual components of the combined method or establish novel-method sufficiency.

One complete 24-example Agent learning batch accepted the memory candidate: validation correct answers increased 2/24 -> 7/24, with one historical loss and ten recoveries. A subsequent router-context candidate was also accepted, giving 8/24 validation accuracy and one further historical recovery; the memory-selector candidate was rejected. Final historical accuracy was 26/72 -> 36/72, with one previously correct case lost and eleven recovered. The original loss allowance of two and output-format threshold of 0.95 were preserved. This is not zero forgetting.

## Protocol and verification

All train24, selection24, development24 and confirmation100 examples were newly allocated from unused h_search source/pixel groups before inference, excluding 2,798 groups from prior and concurrent studies. All 172 groups/pixels are distinct. Confirmation contains 30 yes/no-or-presence cases and 14 of each of five other question types. This is not the original test; patient independence is unverified. The seed is 20261115 and paired intervals use 20,000 bootstrap draws. Development is diagnostic only. The final committed state was frozen before all four confirmation workers started; no best-checkpoint or test-based arm selection was performed.

The design followed prior aggregate failures and inspection of training memory, and was tested on this new panel. Earlier confirmation panels remain historical evidence and were not reused. Results from different panels cannot establish a causal architecture comparison.

G0/GF used no episodic references. GN/GX each retrieved references on 60/100 queries, comprising 102 reference records; source membership was checked against training IDs, with query IDs excluded. Actual backend receipts confirmed GN's multi-image inputs and GX's current-image-only inputs. Their differing image evidence did not yield a demonstrated accuracy difference. This exact-question coverage limits claims about unseen question formulations.

All five workers exited with code zero. The full run took 5,806.0 seconds (96.8 minutes), with 6,085 issued calls and 6,085 completed successful call receipts, and about 44 MB new storage including conservative supervisor overhead. Data isolation, frozen/read-only evaluation, fixed denominators and reference provenance were checked. Before launch, eight gate-boundary checks, five graph tests and four episodic tests passed, followed by real-model visual/text input preflights.

## Release boundary

Executed study source, a path-sanitized protocol template, and aggregate results are included. Runtime dependencies include the existing HCL package, medical adapters, allocation code, model, authorized private data and prior exclusion records. Private images, questions, answers, per-example traces, memberships, memory contents and deployment paths are excluded. This is not a self-contained private-data reproduction bundle. Use neutral environment-configured entrypoints and retain the independent time/storage guard. Do not tune against this completed confirmation panel.
