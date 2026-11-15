# Learned early stopping reduced tool use but failed acceptance

The prespecified candidate was rejected: new acceptance accuracy fell from **9/24 to 8/24**, with zero paired gains and one loss. Historical correctness stayed **38/72**, with zero losses and zero recoveries relative to the frozen predecessor. Format compliance was 23/24 (95.83%). The required current gain of at least one correct answer was not met. No independent confirmation inference was run, and the rejected policy was not deployed.

| Configuration | Training correct /48 | New acceptance correct /24 | Historical correct /72 | Acceptance tool calls |
|---|---:|---:|---:|---:|
| Frozen predecessor, up to two tools | 23 | 9 | 38 | 46 |
| Fixed one-tool diagnostic | 22 | 8 | 38 | 23 |
| Training-derived stopping candidate | Not reevaluated | 8 | 38 | 32 |

The candidate stopped after the first observation on 14/24 acceptance queries. Acceptance tool calls fell from 46 to 32 (30.4%), while accuracy fell by one case. Across the 24 historical VQA anchors, it stopped on 22 queries and used 26 tools versus 48, with unchanged correctness. Other historical task interfaces remained unchanged. This does not establish an acceptable accuracy-cost tradeoff under the frozen accuracy-gain gate. Timing is not evidence of speedup because GPU load differed.

## Training and fixed gate

The same frozen Qwen2.5-VL-7B and learned state from the parent applicability/tool-statistics study were used. There were no weight updates and no new HCL.train calls. The original HCL memory and prompt state stayed fixed. The graph used an actual model finalization after observations, rather than copying a tool answer.

Forty-eight training questions were run with a two-tool maximum and a one-tool maximum. First tool, arguments and observed text matched exactly in 44 pairs; three differed and one had no observation. Sixteen further pairs lacked an eligible visible feature, leaving 28 eligible pairs. A stopping key used only visible question intent, first tool and coarse observation shape (digits, binary or short text); it did not contain the answer identity. Unsupported, uncertain and long observations fell back to the normal loop.

Four keys met the fixed minimum support of four matched pairs and zero training accuracy losses. One key had five examples where both alternatives were wrong; zero measured loss is therefore not evidence of competence. The learned keys and aggregate supports are included in results.json. There was no threshold search.

A single candidate then faced 24 newly allocated acceptance cases and 72 historical anchors. The frozen gate required a net gain of at least one correct current answer, at most two historical correct-to-wrong changes, and at least 95% current format compliance. It rejected the candidate. The fixed one-tool diagnostic was not an alternative candidate chosen after outcomes. The 100 reserved confirmation cases were never evaluated, and the gate was not relaxed to obtain a result.

## Completion, failures and full costs

All three workers and the supervisor exited successfully after 3,813.5 seconds (1.06 hours), within the original four-hour shared clock. Generation calls: 2,101 issued and completed; frozen-feature forwards: 332 (66.6 seconds); total tool calls including preflight: 346. The candidate logged 36 learned stops across acceptance and historical VQA. Final recorded storage was 14,389,248 bytes. Per-role full costs are provided in results.json. Generation seconds summed across workers are not wall time, and heterogeneous concurrent GPU load precludes a controlled speed comparison.

Successful model calls do not mean all Agent episodes succeeded. Each fit arm had one failed training episode and one failed acceptance episode; the candidate had one failed acceptance episode. These remained in the original denominators. The 24 historical VQA episodes completed in each arm; the other 48 anchors used their established direct task interfaces. No failed case was dropped or rerun to improve its score.

## Scope and reproducibility

Training reused the parent 48 cases intentionally. New acceptance24 and reserved confirmation100 excluded 3,190 historical/concurrent source groups and had distinct source/pixel groups. The official test was not accessed; patient independence remains unverified. This is a small mechanism study on a frozen learned state, not validation of a new continual-learning algorithm or a publishable innovation by itself. The negative result directs attention toward the quality of visual evidence and output completeness, rather than further optimizing a stopping rule on these acceptance cases.

The release includes source, boundary checks, the final aggregate collector, path-sanitized frozen protocol and allowlisted aggregate results. Some protocol learning/arm fields are inherited adapter configuration; the action_policy, actual worker code and this report define this experiment. No new two-batch HCL learning was performed here. Runtime depends on the existing HCL/medical adapters, frozen model and authorized data. Medical images, private question/answer records, memberships, per-case traces, learned states, candidate free text and deployment paths are excluded. The current protocol and results remain unchanged.
