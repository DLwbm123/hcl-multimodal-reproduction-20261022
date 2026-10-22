# Medical HCL transfer pilot

Frozen Qwen2.5-VL-7B pilot: classification → polyp localization → visual QA, seed 20261022. Per task: 24 H-search, 24 validation, 24 evaluator-only anchors and 50 test cases. This is a small medical transfer, not an exact reproduction of the original paper model or numerical setting.

Final test: Static **26 / 44 / 20%**, HCL **26 / 40 / 26%**. Descriptive mean: **30.00 → 30.67%**. One VQA memory update was accepted; zero of 36 prompt candidates were accepted. Anchor B=0 did not prevent test forgetting: old-task average forgetting 2pp, BWT −2pp. Localization has an unresolved coordinate-interface confound. Do not interpret this as established overall HCL superiority.

Read [the Chinese final report](REPORT.zh-CN.md) and [machine-readable results](results/final_validated_results.json).

## Published scope

`src/medical_hcl` contains preparation/strict evaluator/bridge code; `tests` and `scripts` contain CPU checks and metadata preparation. Use the pinned HCL source at repository root (historical base commit d0e5ecd). Model/data locations are runtime arguments or environment variables; private server configuration is excluded. Reuse legally obtained Kvasir-family assets; images, labels, patient data, model weights and individual prediction logs are not distributed. Frozen anonymous case IDs are included for auditing membership; reconstructing their source linkage requires the private source-preparation manifest. Patient independence is unverified.

`coordinate_interface` and `template_contract` are **isolated preparation only**, never loaded by this completed pilot. Coordinate geometry/caller checks passed, but fixed synthetic model-accuracy acceptance failed (0/4); no new paired experiment was started. Template structural repair passed 24×12 development rendering combinations and eight tests; it does not establish a performance gain. Original failed candidates are retained. No repaired candidate reuses its original performance score.

CPU checks from repository root:

```sh
PYTHONPATH="$PWD:$PWD/medical_transfer/src" HCL_UPSTREAM_ROOT="$PWD" python -B -m unittest discover -s medical_transfer/tests
PYTHONPATH="$PWD:$PWD/medical_transfer/template_contract" python -B -m unittest discover -s medical_transfer/template_contract
# The coordinate caller tests additionally require PyTorch and Pillow; no real weights are used.
PYTHONPATH="$PWD:$PWD/medical_transfer/src:$PWD/medical_transfer/coordinate_interface" python -B -m unittest discover -s medical_transfer/coordinate_interface
```

The published code provides no automatic experiment launcher. Do not hot-patch an existing run. The opt-in candidate repair and coordinate boundary require fresh pipeline state and separately declared future protocols/budgets. Performance gates and failure denominators remain unchanged.
