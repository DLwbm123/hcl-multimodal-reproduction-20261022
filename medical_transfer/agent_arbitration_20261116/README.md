# No training-supported contract-only override; reserved evaluation remained unopened

The frozen incumbent context router scored **25/48** on training, versus **23/48** for contract-only observations. There were zero contract-only gains and two losses, one in color and one in location. No visible-intent group met the prespecified support >=4, net gain >=1 and zero-loss rule. The learned override list was empty, so the supervisor ended the experiment **without acceptance or confirmation evaluation**. The existing policy remained unchanged.

| Visible intent | Training support | Incumbent correct | Contract-only correct | Gains / losses |
|---|---:|---:|---:|---:|
| Presence | 14 | 11 | 11 | 0 / 0 |
| Count | 8 | 7 | 7 | 0 / 0 |
| Color | 8 | 1 | 0 | 0 / 1 |
| Location | 8 | 2 | 1 | 0 / 1 |
| Other | 10 | 4 | 4 | 0 / 0 |

These are training results, not independent attribution evidence. They do not prove that independent visual observations caused the preceding three-point confirmation gain, nor that contract-only behavior is generally worse. The unanswered source-versus-format confound remains a limitation.

## Fixed hypothesis and execution

The need to separate output-format contracts from independent tool observations was recorded before viewing the preceding routing confirmation. This study reran both training arms on the same 48 examples rather than assembling predictions from older runs. Model weights and the parent's raw/abstract/prompt state stayed fixed; no HCL.train was called.

The incumbent used independent observations plus output contracts for its training-supported color/location routes, and original guided tools for other intents. Contract-only retained original tool guidance while adding the same training-derived color/location output contracts to tool and final-answer prompts. A proposed policy could replace an incumbent route only when its paired training evidence met the fixed rule. Empty support ended the study. A contract-only ablation was never eligible as a fallback candidate.

## Preserved data and costs

Because a metadata-only allocation check could no longer form a new balanced acceptance panel, the last untouched 100-case reserve from the rejected action study was partitioned before inference: four cases per six question types for acceptance24, with the remaining76 reserved for confirmation. The original action gate was rejected and no confirmation workers had run. All 100 source/pixel groups stayed distinct and disjoint from later study panels. Neither partition was used for fitting, and this experiment opened neither for model evaluation. Preserve this fixed split; do not reshuffle to obtain a favorable result. The official test was untouched and patient independence remains unverified.

Both workers exited successfully after 861.34 seconds (14.36 minutes), within the original three-hour clock. Costs including preflight: 790 generation calls issued/completed, 128 feature forward passes (12.85 seconds), 196 tools, 98 completed and two failed Agent episodes. Each training arm had one failed episode retained in its 48-case denominator. Recorded storage was 5,906,432 bytes. Shared GPU load prevents controlled timing claims.

Final verification reconstructed all training group results, checked empty-candidate termination and absence of eval/confirm workers, validated actual source choices and training-only references without self-reference, and checked read-only state/frozen weights and full cost accounting. Medical images, private labels/questions, memberships, trajectories, learned states and feedback vocabularies are excluded from this public release.

The training spatial diagnostic is descriptive: the eight color queries used 13 whole-image VQA, two region-inspection and one classification calls; the eight location queries used 16 localize-region calls. A bounded localization-then-crop mechanism may be tested from training evidence, but this diagnostic does not establish a cause of errors or justify bypassing independent acceptance.
