# HCL multimodal reproduction — Qwen2.5-VL-7B

[新的12小时自主实验窗口](medical_multimodal/AUTONOMY_12H_20261025.zh-CN.md)：2026-10-25 22:54至10-26 10:54（北京时间），恢复有界自主方法实验，替代此前只收尾安排；v11原协议与截止不变，E1审核前置条件仍未满足。

[最新医学 Agent 结果汇总（2026-10-25）](medical_multimodal/CURRENT_RESULTS_20261025.zh-CN.md)：v5–v10 共848实际回合、2629次订阅请求，尚无可信、可重复、可归因的记忆学习收益；v10名义+1未使用经验。汇总区分数据问题、受限方法实现与未证实的机制解释，v11仍在运行。

A resource-bounded reproduction of [Harness Continual Learning](https://arxiv.org/abs/2608.19013v2), based on [official code](https://github.com/RL-MIND/Harness-Continual-Learning) commit `a6e5a01536ac8d42ca28f0a82efa223869c8c48d`. **Not an exact paper-setting or paper-number replication.** The Chinese experimental report is [REPORT.md](REPORT.md); license/provenance are in [UPSTREAM.md](UPSTREAM.md).

| Method | This run final mean (0–100) | Avg. forgetting (pp) | Paper Table 2 mean |
|---|---:|---:|---:|
| Raw Zero-shot | 60.13 | — | 46.77 |
| Static Harness | 53.08 | — | not reported |
| Stability-HCL, recovery protocol | 54.46 | 0.00 | 68.92 |
| Plasticity-HCL, recovery protocol | 51.25 | 3.38 | 67.96 |

Both strict HCL runs failed on an invalid memory-selector role. Separate, declared failed-sample-accounting recoveries completed; strict failures and their costs remain in the report. HCL did not outperform raw zero-shot here. The backbone is Qwen2.5-VL-7B, not paper Qwen3.6-27B; counts are 50/10/100, not 250/50/500. No post-test tuning, extra seeds or weight updates were performed.

## Contents

- Latest evidence-guidance diagnostic: [v8 negative result](medical_h0/evidence_guidance_v8/REPORT.zh-CN.md). Original prompt 5/12 vs added checklist 3/12; not adopted, no HCL.

- Latest medical-tool input ablation: [v7 report](medical_h0/input_ablation_v7/REPORT.zh-CN.md). Autonomous navigation 3/12 vs all-raw-facts-visible 9/12; independent mapping 6/6 numeric but 4/6 trusted. All 30 synthetic episodes audited, no HCL launch.
- Prior component diagnostic: [v6 report](medical_h0/component_diagnostic_v6/REPORT.zh-CN.md). Submission 6/6, navigation 2/6, mapping unattempted; the historical result is unchanged.

- `hcl/`: official core package plus documented fixes; MIT license retained.
- `configs/native_*.template.json`: public templates with model/data-root placeholders.
- `protocols/`: frozen native/recovery protocols and upstream-relative patch.
- `results/`: aggregate metrics/costs, public sample IDs, test summary and CSV comparison.
- `scripts/`: original preparation utilities and resumable download/builder scripts.
- `r000.py`, `r001.py`, `r002.py`: neutral preparation, audited arm and guarded fresh-run entrypoints.

No data, model weights, source PDF, per-example prediction logs, private configuration or checkpoints are distributed.

## Offline verification

Python 3.12 was used. Core offline tests work without downloading the model:

```sh
python -m unittest discover -s tests -v
python scripts/verify_release.py
```

The experimental server passed all 29 tests. A clean local package check passed 27 with 2 COCO-dependency skips. Install the evaluation extras to run those tests too.

## Prepare an independent replay

This is **not a command to resume or reset the completed historical experiment**. Use a new, dedicated data-disk run root. Keep all new environments/cache/logs under that root; an existing model may be reused read-only. On Linux, provide four GPUs each with at least 60,000 MiB free. Never stop someone else's processes.

1. Place this checkout at `$REPRO_ROOT/code`. Set `MODEL_PATH` to an existing local Qwen2.5-VL-7B-Instruct folder. Keep Java 21 on PATH. Create the environment under `$REPRO_ROOT`, then install `requirements-reproduction.txt`. Use the Torch CUDA build appropriate for your GPU; the original run used 2.7.1+cu128.
2. Obtain official COCO2014 train/val images and annotations, VQAv2 train/val questions/annotations, and RefCOCO-UNC according to their respective terms. `scripts/fetch_original_data.py` downloads the COCO/VQA archives; it requires an external guard if used, and does **not** fetch RefCOCO. Sources and archive names are in that file and `scripts/build_multimodal_local.py`. Original RefCOCO source is currently represented by the authors' archive backup listed in the manifests; supply `raw/refcoco.zip` yourself.
3. Required filenames in `raw/`: `train2014.zip`, `val2014.zip`, `annotations_trainval2014.zip`, `refcoco.zip`, `vqa_questions.zip`, `vqa_annotations.zip`, `vqa_train_questions.zip`, `vqa_train_annotations.zip`. Extracted COCO JPGs share the flat `images/` directory. Keep ZIPs. Data is around 41 GB with retained archives.
4. From the checkout, run `python r000.py`. This prepares the old pilot IDs for exclusion and the native 50/10/100 dataset, asserts the exact public sample IDs, and generates private `configs/runtime_*.json` files. No model is loaded. An existing budget start is preserved; never delete/reset it to extend retries.
5. Choose strict or failed-sample-accounting mode before execution:

```sh
# Set REPRO_ROOT, MODEL_PATH and the environment's Python/PATH before these commands.
# Default FAIL_CLOSED=0 reproduces strict exception handling; invalid roles can terminate an arm.
export GPU_IDS='[0,1,2,3]'
export FAIL_CLOSED=1
nohup python -u r002.py > "$REPRO_ROOT/supervisor.log" 2>&1 &
```

`FAIL_CLOSED=1` enables the declared policy for both HCL arms: after the original two selector schema attempts fail, score that entire sample as failed, with no final answer and no additional retry. It never maps invalid roles, relaxes gates or drops failed samples. A **fresh** fail-closed replay is not byte-identical to the historical strict-prefix-then-recovery trajectory reported here; no historical checkpoints are shipped and deterministic execution is not guaranteed across hardware/software.

The fresh replay supervisor runs exactly four independent arms, no retries and no follow-on experiments. It guards the original shared 23.5-hour early stop, 47 GB total/39 GB nondata conservative thresholds (1 GB cache reserve), measuring every 15 seconds; all descendants have neutral command lines. Nondata excludes only the declared raw/images/generated dataset directories. Existing artifacts cause refusal rather than overwrite. The public supervisor was syntax/release checked, not used to launch another GPU experiment during publication. Inspect `ps` and `nvidia-smi` after launch, and check EXIT/FAILURE plus complete matrices rather than treating a marker or idle GPU as proof.

For an individual audited arm, set `RUN_CONFIG` to its generated runtime config and invoke `python r001.py` **only under an equivalent shared time/storage guard**. Caption requires functioning Java; verify `caption_metric_source=pycocoevalcap`, not fallback.

## Metrics and interpretation

Scores are official-code task-native primary metrics: detection micro-F1@IoU0.5 (COCO AP is separately retained), caption BLEU4/METEOR/ROUGE-L/CIDEr composite, grounding Acc@IoU0.5 and VQA consensus. Final mean is unweighted on the 0–100 scale. Average forgetting follows Appendix C.2 Eq.11; BWT is separately reported. Table 2 does not fully specify all metric implementation details, so complete metric equivalence to the paper remains unverified.

The public release contains aggregate receipts rather than raw model outputs. Calls include strict failures and recovery replay: 15,887 for the native comparison, 19,619 known including pilot/preflight/smoke/mistaken auxiliary calls. Some auxiliary timing was not instrumented and no token cost was measured. See REPORT.md for limitations and negative results.

## Medical transfer pilot (2026-10-22)

Completed small medical transfer: Static 30.00%, HCL 30.67% descriptive final mean, with localization regression and unresolved coordinate confound. See [medical transfer code and validation](medical_transfer/README.md) and the [Chinese final report](medical_transfer/REPORT.zh-CN.md). Isolated interface/template repairs were not deployed into this run; no follow-up paired experiment was started.

## Bounded H0 executable gate v5

The [18-target H0 v5 report](medical_h0/executable_gate_v5/REPORT.zh-CN.md) records a completed, failed gate: 0/18 correct submissions, with two evidence-supported computations left unsubmitted. No HCL was started. The source, protocol, aggregate diagnostics and resource audit are included; private clinical fixtures and raw traces are excluded.

- [v9 显式操作数—来源绑定诊断](medical_h0/operand_binding_v9/REPORT.zh-CN.md)：16/16 次任务完成；可信完成 R 2/8、I 1/8，MME 均 0/6。4 条 I 剂量声明来源完整，但先被药名转换错误阻断；未采用，未启动 HCL。

- [v10 参数级反馈与有界修复](medical_h0/bounded_repair_v10/REPORT.zh-CN.md)：CPU 反事实回放通过；四组恢复均 0/6，32 次生成全部直接弃答。门槛失败，跳过确认集并结束此路线；未启动 HCL。

## Current medical multimodal route

[MedRAX native qualification](medical_multimodal/medrax_v1/REPORT.zh-CN.md) completed: 4/12 correct, all four real tools executable, but only 1/12 autonomous tool-use episodes; the frozen gate failed and no HCL updates were run. A [single paired consultation diagnostic](medical_multimodal/medrax_tool_consult_v2/README.md) completed on 12 fresh cases: native 6/12 versus consultation 4/12, with two losses and no gains. The prompt-adaptation branch is closed; no HCL launch. Historical H0 results remain unchanged.

MedRAX fixed tool-evidence diagnostic is complete: native 10/12, classifier seed 11/12, expert seed 5/12; the local evidence-adaptation route is closed. [Report](medical_multimodal/medrax_tool_value_v3/REPORT.zh-CN.md).

The distinct [native ReAct visual-tool qualification](medical_multimodal/react_vision_v1/REPORT.zh-CN.md) also closed: 4/12 correct, 6/12 valid finals, zero autonomous visual-tool calls, despite all direct tool checks executing. Current local bases have not qualified for HCL; no retention or HCL efficacy claim.

## 2026-10-24 twelve-hour progress report

[过去 12 小时进展报告](medical_multimodal/reports/2026-10-24_12h/REPORT.zh-CN.md) summarizes four sealed medical-agent experiments (84 episodes, 48 development case groups), failed qualification decisions, full planner counts, the subsequent waiting interval, and a proposed next direction. No new HCL efficacy result is claimed.

## Sol High subscription medical-agent qualification

[Author-aligned six-tool qualification](medical_multimodal/medrax_subscription_v4/REPORT.zh-CN.md) completed: 12/12 episodes and autonomous tool round trips, 11/12 correct. An initial image-path transport failure and all retry costs are retained (113 total reserved subscription calls). Tool outputs include low-information and malformed responses; operational qualification does not establish causal tool benefit or HCL efficacy.

[首轮 Sol High 医学记忆持续学习](medical_multimodal/medrax_hcl_v5/REPORT.zh-CN.md)已完成：160回合、488调用，总正确39/48未提升，A退化1例、B更新被拒绝；抽象记忆未进入检索，联合目标失败。

[v11视觉经验结项](medical_multimodal/medrax_visual_memory_v11/REPORT.zh-CN.md)：完整第一轮H0 22/24、文字20/24、图文21/24、错配21/24；原6小时截止，106/192回合启动，第二轮不完整。没有观察到图像经验增益，限流与全部失败成本保留。

[v12初答后经验核查已启动](medical_multimodal/medrax_postdraft_memory_v12/README.zh-CN.md)：6个既有病例、三臂两轮36回合，对照原流程与等额通用核查；220次调用/原6小时上限，初答已正确的名义差异不算经验修复。

[v12结项负结果](medical_multimodal/medrax_postdraft_memory_v12/REPORT.zh-CN.md)：36/36完成，经验组两轮均与H0同为4/6，0初答修复；137次请求全部ok。

[v13训练反馈工具查询已启动](medical_multimodal/medrax_tool_query_v13/README.zh-CN.md)：24个既有B训练来源生成一次查询程序，三臂6病例两轮36回合，固定与学习查询使用同一工具/图像/调用次数；220次请求/原6小时上限。
