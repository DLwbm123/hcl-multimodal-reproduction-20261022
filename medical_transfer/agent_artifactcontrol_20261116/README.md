# Random artifact names removed from model-visible observations

The original tool interface exposed random storage filenames in six controller/finalizer calls, totaling seven identifier occurrences. Both cleaned runs exposed **zero**, while all four original private artifact files and their raw receipts remained available in each arm. This validates an engineering input-invariance fix, not a performance improvement.

| Training arm | Correct / fixed denominator | Model-visible artifact occurrences | Private artifacts retained |
|---|---:|---:|---:|
| Original raw results | 25/48 | 7 | 4 |
| Clean A | 25/48 | 0 | 4 |
| Clean B | 25/48 | 0 | 4 |

Raw-to-clean had zero correctness gains/losses and one changed answer string. Clean A/B had identical answers on all 48 cases. Each arm retained one failed Agent episode in its denominator. There is no independent generalization or continual-learning claim, and no policy was fitted or selected based on these scores.

## Exact engineering change

For known inspect_region and segmentation tool-result JSON dictionaries, model_observation omits only the top-level artifact, mask_artifact and overlay_artifact storage fields from the observations serialized into controller/finalizer prompts. Clinical observation text, status, original-pixel geometry and mapping are preserved. Other tools, non-JSON results and non-dictionaries pass through unchanged. Original ToolMessages, private traces and artifact files remain intact. The experiment uses an explicit strip_artifact_names switch so the unchanged raw control remains reproducible.

This issue was identified through a read-only audit of a previous unchanged-route acceptance discrepancy: semantic tool results agreed, random filenames differed and final answers differed. That establishes unequal model-visible inputs, not causality of the answer discrepancy. The previous acceptance/confirmation cases were never rerun or used to fit this fix. Tests directly verify invariance under different storage filenames and preservation of semantic fields and private files.

## Frozen training-only validation

Prespecified order on GPU4: raw48, cleanA48, cleanB48, fresh process per arm, seed2026111608. Same incumbent routing, frozen model and learned HCL state, training references, clinical prompt wording and output contracts. Only original training48 were copied. No HCL.train, skill fitting, acceptance, confirmation, winner selection or deployment occurred. Existing held-out panels remain consumed; official test was unused and patient independence is unverified.

Six interface/geometry tests and five original graph tests passed before launch. All three real-model preflights, workers and the supervisor succeeded; final verification checked fixed denominators, training-only references, read-only/frozen state, model-prompt exposure counts and actual retained files. The two clean runs enforced zero random identifiers at every controller/finalizer generation.

Elapsed time: 1,412.56 seconds (23.54 minutes), under the original two-hour deadline. Recorded storage: 8,003,584 bytes. Including preflights: 1,185 successful generations, 192 feature forwards (11.03 seconds), 294 tool executions and 300 controller calls. Shared-GPU timing is descriptive. Clean agreement on this reused training panel does not prove universal determinism.

Published: executed code, sanitized protocol, aggregate results, completion and verification receipts. Excluded: private images/crops, text/labels, memberships, per-case traces/boxes, detector text, learned states and credentials.
