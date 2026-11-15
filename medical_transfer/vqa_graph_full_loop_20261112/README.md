# Full medical VQA learning through LangGraph: completed negative learning result

This study ran the full two-batch HCL feedback, memory proposal, prompt-candidate evaluation and commit/rollback workflow through the bounded LangGraph agent. All VQA cases used the graph; classification and localization historical anchors kept their original direct, coordinate-safe interface. No model parameters were updated.

The main outcome was **no accepted learning update**: both memory proposals and all eight prompt-component candidates were rejected. The final fresh Agent matched its initial state on every confirmation example: **25/200 before and 25/200 after**, with zero gains and zero losses. The fixed-six-example Agent performed worse, but that does not turn the unchanged learning arm into a learning success.

| Arm | Description | Correct / 200 | Graph episodes completed / 200 |
|---|---|---:|---:|
| G0 | Initial empty-memory Agent | 25 | 125 |
| GF | Agent with fixed six examples | 2 | 54 |
| GT | Agent with the prior direct-trained memory | 68 | 182 |
| GN | Fresh Agent after two full learning batches | 25 | 125 |
| D0 | Direct VLM, no memory | 60 | N/A |
| DT | Direct VLM, same prior memory as GT | 78 | N/A |

Both memory candidates improved the 48-case selection panel from 1 to 9 correct, but each lost three previously correct historical cases, exceeding the allowance of two. The second also failed the current-format threshold. Both were rolled back. Fresh development performance was 8/80. Frequent malformed controller actions were retained as failed cases in every fixed denominator.

An additional, explicitly **post-result descriptive comparison** shows that the previously learned state transfers to the Agent: GT versus G0 has 49 gains and 6 losses, +21.5 percentage points, paired bootstrap 95% interval [15, 28]. This is not evidence that the current Agent-learning run accepted new knowledge. GT scored below the same memory on the direct path (68 versus 78); it scored above direct no-memory by 4 percentage points, with interval [-2, 10.5]. All contrasts and negative evidence are retained in `results.json`.

## Protocol and repair

Training48, selection48 and development80 were reused from the earlier direct study. The new 200-case confirmation panel was reserved from previously unused h_search source/pixel groups before model outputs; all prior memberships were excluded. The train_replay pool could not support all six question types, so the source was changed before model execution. This is not the original test, and patient independence is unverified.

The first confirmation attempt stopped when an Agent requested a 13-by-47-pixel crop unsupported by the pinned image processor. The tool now returns an explicit input-error observation for dimensions below 28 or aspect ratio above 200. Legal inputs and backend failure propagation are unchanged. All confirmation arms restarted consistently; original attempts, original clock and issued-call counts were retained. Training and the committed state were not rerun or retuned. Targeted boundary tests passed.

All seven workers (one learning, six confirmation) exited successfully. The full run, including interruption and recovery delay, took 43,967 seconds and used about 69 MB of new storage. There were 14,317 issued model calls and 14,315 completed call receipts; two in-flight calls were terminated by the failure supervisor. `results.json.total_cost_including_interrupted_attempt` includes the failed attempt; the per-confirmation-arm `costs` describe the final attempts only.

## Contents and boundaries

Source, aggregate results and a protocol template are provided. The runtime requires the existing HCL package, medical bridge, coordinate/semantic/task-scope adapters and authorized private assets. Deploy neutral entrypoints through environment variables; retain the independent timeout/storage supervisor. Private images, questions, answers, memory contents, per-example traces and deployment paths are intentionally omitted. This is not a self-contained private-data reproduction bundle.

The follow-up investigation separates controller action decisions from final medical answering; it is a different experiment and does not alter these results.
