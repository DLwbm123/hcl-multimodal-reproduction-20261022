# Grounded crop skill passed acceptance but did not improve independent confirmation

The training-supported location skill scored **44/76**, versus **45/76** for the unchanged incumbent, on the once-only confirmation panel: zero gains, one loss, -1.32 percentage points; paired bootstrap 95% interval [-3.95, 0.00], exact McNemar p=1.0. The always-eligible macro also scored 44/76 with identical correctness. These results do not support a generalization benefit or a validated continual-learning contribution.

| Stage | Incumbent | Proposed / training macro | Always-eligible macro |
|---|---:|---:|---:|
| Training | 25/48 | 26/48 | 26/48 |
| Acceptance | 12/24 | 13/24 | 12/24 |
| Historical anchors | 38/72 | 38/72 | 38/72 |
| Independent confirmation | 45/76 | 44/76 | 44/76 |

## Mechanism and attribution

A frozen Qwen proposes the visible target union box in normalized integer coordinates. Strict schema/boundary validation converts it to original pixels, adds 10% margin on each side, enforces a 28-pixel minimum and clamps to the image. A second tool reads the actual crop, or falls back to whole-image VQA if the proposal is absent/uncertain/invalid. Finalization still sees the original image, learned HCL guidance, training references and output contracts. The hidden proposal primitive is unavailable to the ordinary controller. At most two tool observations are allowed; macro execution replaces controller choices and is explicitly logged as prescribed, not autonomous planning.

Only training pairs determine skill applicability: support >=4, net gain >=1, zero losses. Location alone qualified (2 to 3 correct out of 8); color remained 1/8. The only training gain occurred after an absent proposal and whole-image fallback. Five of 16 training proposals produced actual successful crops; ten were absent and one invalid. Crop-path training cases had no gains or losses.

The frozen acceptance gate passed (net +1/24, zero historical losses, format 100%), but post-run attribution found its sole gain in an **unchanged non-macro route**, not the four treated location examples (0/4 both). This is unexplained execution variation, not evidence that the skill improved acceptance. Do not retroactively alter the gate or relabel that gain as a treatment effect. Confirmation had ten treated location episodes: five successful crops, four absent fallbacks, one invalid fallback. Its sole loss occurred on a successful-crop path. Valid geometry establishes execution, not semantic localization accuracy; path subgroups are descriptive and not causal evidence.

## Protocol, data and limitations

Model weights and the parent's learned raw/abstract/prompt state remained frozen. No HCL.train was called. Existing color/location independent observation and output contracts remained fixed. The last unused action reserve was passed through arbitration and inherited unchanged as selection24/confirmation76 (partition seed 2026111605). **Both partitions are now consumed.** The 76 cannot be reused as fresh selection or held-out confirmation; do not return to the earlier source run to claim it was unopened. The official test was untouched and patient independence is unverified.

The executed model/bootstrap seed was 2026111606. The frozen protocol contains inherited descriptions referring to earlier arms, gates and CI seed2026111603; these are stale descriptive fields, preserved for transparency. The executed grounding entrypoints use the top-level seed, location-only learned policy, selection24/historical72 gate, and 76-case confirmation as described above. Scientific interpretation follows actual execution and receipts, not those inherited descriptions.

All eight workers exited successfully; final verification reconstructed denominators, training policy, gate and pairing, checked frozen/read-only evaluation, training-only references without self-reference, macro selection and two-tool budgets, and original-pixel crop mapping. Each training arm retained one failed Agent episode in its 48-case denominator. Total costs including preflight: 4,111 successful generation calls, 821 feature forwards (74.24 seconds), 952 tool executions, 476 completed and two failed graph episodes. Wall time 3,966.36 seconds (66.11 minutes), recorded storage 40,243,200 bytes; original three-hour deadline respected. Shared-GPU timing is not a controlled speed comparison; controller reduction and view changes are confounded.

Published: executed source, sanitized protocol, aggregate results and attribution. Excluded: medical images/crops, private questions/labels, memberships, per-case traces/boxes, raw detector text, learned memory and credentials.
