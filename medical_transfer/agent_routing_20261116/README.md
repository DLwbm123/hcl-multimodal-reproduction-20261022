# Training-derived context routing: three gains, no losses, but uncertain confirmation benefit

On the prespecified 100-case confirmation panel, the frozen predecessor scored **48/100**, the training-derived router **51/100**, and unconditional independent observations **46/100**. The primary routed-versus-predecessor comparison had three gains and zero losses: **+3 percentage points, paired bootstrap 95% interval [0, 7], exact two-sided McNemar p=0.25**. This small panel does not establish a reliable improvement. The secondary comparison against unconditional independence had nine gains and four losses: +5 points, interval [-2, 12], p=0.26685. Secondary results are descriptive, not multiplicity-adjusted.

| Arm | New acceptance /48 | Historical /72 | Confirmation /100 | Confirmation independent routes |
|---|---:|---:|---:|---:|
| Frozen predecessor | 15 | 38 | 48 | 0 |
| Unconditional contract + independent observations | 17 | 37 | 46 | 100 |
| Training-derived routing (sole candidate) | 16 | 38 | 51 | 28 |

The candidate passed the frozen acceptance gate with one gain and zero losses, 100% format compliance, and no historical losses or recoveries. The larger acceptance total of the unconditional ablation did not cause an arm switch. All 300 confirmation Agent episodes completed; every arm used 200 tools and 799 generation calls. There is no tool-cost reduction.

## What was learned

Two full training counterfactuals used the same 48 previous training questions and same frozen learned HCL state. Original guidance scored 23/48 and unconditional independence 18/48. One failed Agent episode in each arm remained in the denominator. Visible question intent, not dataset metadata or answer identity, defined policy groups.

A group selected independent observations only with support >=4, at least one net training gain, and zero training losses. Color and location qualified with eight examples and one gain each. Presence lost three cases, count had one gain and two losses, and other lost three. These groups fell back to original guidance. This differs from manually picking count based on a preceding acceptance panel; count was excluded by actual training evidence.

At inference, routed color/location questions use independent tool requests and training-derived output contracts. All other questions retain the original guided tool path and no new output contract. The router switches both tool guidance and final-answer contract, so this study cannot isolate their contributions. Contract-only confounding was recorded before confirmation results were viewed. The remaining scientific question is whether independence adds value beyond output formatting.

Raw/abstract memory, learned prompts and model weights remain fixed. This fits an external routing policy candidate, not a new call to HCL.train, an end-to-end learned controller, or validated multi-stage continual learning. Training fitting is in-sample on a state that has already encountered these 48 examples; new acceptance and confirmation are therefore essential. The empirical intent rule is not a calibrated reliability estimate.

## Data amendment before inference

Version 1 attempted all-new acceptance48 plus confirmation100, but allocation failed for the count stratum before any membership was written or any model call. Version 2, frozen before inference, allocated 48 new acceptance cases excluding 3,462 prior used/reserved source groups and transferred the exact 100-case confirmation panel from the preceding evidence study. That predecessor had rejected its candidate and had no confirmation workers, predictions or scores. The transferred panel was never used for training or acceptance and remained disjoint from the new acceptance groups and pixels.

Thus these are 48 newly allocated cases plus 100 previously reserved but unevaluated cases, not 148 newly allocated cases. No revealed confirmation was reused for fitting. The official test was not accessed; patient independence remains unverified. Seed 2026111604, 20,000 bootstrap draws, thresholds and role definitions were frozen before inference. The confirmation mix is a custom six-type panel, not a population-weighted clinical sample.

## Completion and verification

All eight workers and the supervisor exited successfully in 4,916.3 seconds (1.37 hours), within the original four-hour clock. Total costs including preflight and failed episodes: 5,395 generation calls issued/completed, 956 feature forward passes taking 91.67 seconds, 1,240 tools, 620 completed and two failed Agent episodes. Recorded storage was 29,519,872 bytes. Total call seconds are summed across concurrent workers and must not be interpreted as wall time or a controlled speed comparison.

Final checks verified role membership and fixed denominators, read-only learned state and frozen weights, training-only references with no self-reference, reconstruction of the learned policy from actual training outcomes, and inference route choices for every training/acceptance/confirmation question. Historical route counts were also recorded; per-question route reconstruction was limited to the known-query map, not claimed for all historical anchors. No failed episode was excluded or rerun to improve scores.

The source, path-sanitized protocol, aggregate results and audit utility are public. Medical images, private question/answer records, case memberships, trajectories, learned states and exact feedback vocabularies are excluded. Existing HCL/medical adapters, model and authorized data are required. This is a bounded mechanism study with a weak positive result, not evidence that the full method is ready for a top-tier contribution.
