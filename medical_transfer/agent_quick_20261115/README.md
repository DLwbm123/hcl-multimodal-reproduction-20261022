# Role-separated LangGraph Agent: accepted memory, no independent accuracy gain

A complete single-batch HCL learning run accepted a new memory state, but its independent 100-case confirmation did not improve: **18/100 before learning, 17/100 after learning**. Fixed six-example guidance scored 25/100. All three arms completed 100/100 Agent episodes. Execution completion is not answer correctness.

| Arm | Definition | Correct / 100 | Completed / 100 |
|---|---|---:|---:|
| G0 | Initial empty-memory Agent | 18 | 100 |
| GN | Final committed state after one 24-example learning batch | 17 | 100 |
| GF | Same Agent with six fixed training examples | 25 | 100 |

The predeclared primary contrast GN-G0 is -1 percentage point: 5 gains, 6 losses, paired bootstrap 95% interval [-7, 5], exact two-sided McNemar p=1.0. The secondary GN-GF contrast is -8 points: 2 gains, 10 losses, interval [-15, -2], nominal p=0.0386 without multiplicity adjustment. This is a negative generalization result, not a successful learning-performance claim.

## What changed and what was learned

The real LangGraph execution loop separates controller action decisions from medical final answers: process -> execute -> process -> finalize. The controller receives the question, tool schema and observations; learned HCL guidance reaches the visual-language tools and final answer node. There are at most two tool executions and one action-format retry. The model and tools share frozen Qwen2.5-VL-7B weights.

One complete 24-example training batch produced and accepted a memory candidate: selection correct answers increased from 3/24 to 8/24, with two previously correct historical cases lost and one recovered. This met the frozen historical-loss allowance of two. The deployed memory contains 10 raw and 2 abstract entries, 72 historical anchors and no learned skills. Both prompt-component candidates were rejected. The harness prompt-version counter remained zero; this does not negate the separately recorded accepted memory update.

Final historical retention was 26/72 initially and 25/72 after learning, with 2 losses and 1 recovery. This is bounded forgetting, not lossless retention. Selection improvement did not transfer to the independent confirmation panel.

## Frozen protocol and evidence boundaries

This expedited diagnostic ran alongside a separate, larger two-start experiment and does not replace it. Training24, selection24, development24 and confirmation100 were allocated before inference from unused h_search source/pixel groups. All 172 groups/pixels were unique and disjoint from 2,626 excluded groups, including the larger concurrent study. The original test was untouched; patient independence is unverified. Confirmation used 30 yes/no-or-presence cases and 14 cases for each of five other question types. Development was diagnostic only, with no state selection. GN is the final committed state, not a selected best checkpoint. Seed: 20261115; paired bootstrap: 20,000 draws.

The original frozen protocol prose retained a stale `1/48` selection denominator from its parent study. Actual allocation, structured configuration, runtime and metrics all use 24 cases, and the enforced threshold is an absolute correct-answer gain of at least one. No gate was changed during the run. Unused resume-initialization metadata was also inherited; this experiment used fresh empty memory only. The template preserves that metadata and these discrepancies are disclosed rather than retrospectively rewriting the frozen record.

All four workers exited with code zero. Total wall time was 5,795.7 seconds (96.6 minutes), within the three-hour clock. Total model calls: 5,324 issued and 5,324 completed receipts, with no interrupted calls. New storage was about 43.3 MB including the supervisor's conservative overhead. Each confirmation arm had one recoverable action-format error, no failed episodes, and unchanged evaluation state with frozen weights. Eight gate-boundary checks and five graph tests passed before launch.

## Contents and reproduction limits

`code/` contains the executed study source, `protocol.template.json` replaces deployment paths with placeholders, and `results.json` retains aggregate results, gate decisions, costs and completion evidence. The runtime requires the existing HCL package, medical adapters, allocator, authorized private data and prior source/pixel exclusion records. The frozen model is a separate shared dependency. This is not a self-contained private-data reproduction bundle.

Private images, questions, answers, memory contents, per-example predictions/traces, membership identifiers, deployment paths and credentials are excluded. Launch with neutral entrypoints and environment variables while retaining the independent deadline/storage supervisor. Do not use this confirmation panel to retune the method.
