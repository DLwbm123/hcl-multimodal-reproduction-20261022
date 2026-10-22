# HCL 多模态持续学习复现实验报告

日期：2026-10-22。状态：本次受限对照推理已全部结束；不继续调参或启动后继实验。两个HCL的原严格版本失败，以下完整结果来自单独记录的fail-closed恢复协议。

## 结论与证据等级

**实现链路在恢复协议下跑通；论文设置未完全匹配；论文的总体性能增益未复现。**

Stability恢复版最终平均54.46，高于Static Harness的53.08（+1.37个百分点），低于直接Zero-shot的60.13（−5.67）。Plasticity恢复版为51.25，低于Static（−1.84）和Zero-shot（−8.88）。Stability的测试平均遗忘为0，Plasticity为3.38个百分点，支持本次有限样本中更强约束与较低遗忘的对应关系，不能由单次小样本运行证明方法普遍有效或因果贡献。两个HCL的VQA性能明显低于Zero-shot。

## 1. 原值与本次结果

论文原值逐项核对用户提供PDF第8页Table 2；没有将附录消融Table 7或其他底座实验混入主表。[论文](https://arxiv.org/abs/2608.19013v2)。所有列换算到0–100，遗忘单位为百分点。

| 方法 | 来源 | Detection | Caption | Grounding | VQAv2 | Final Mean↑ | Avg. Fgt.↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | paper | 37.52 | 21.10 | 43.00 | 85.47 | 46.77 | — |
| Zero-shot | this_run | 45.67 | 34.84 | 92.00 | 68.00 | 60.13 | — |
| Static Harness | this_run | 53.77 | 36.90 | 84.00 | 37.67 | 53.08 | — |
| Stability-HCL | paper | 65.34 | 39.41 | 91.60 | 79.33 | 68.92 | 0.22 |
| Stability-HCL | this_run_recovery | 60.16 | 38.01 | 84.00 | 35.67 | 54.46 | 0.00 |
| Plasticity-HCL | paper | 64.14 | 37.31 | 90.60 | 79.80 | 67.96 | 0.81 |
| Plasticity-HCL | this_run_recovery | 51.07 | 36.59 | 83.00 | 34.33 | 51.25 | 3.38 |

Table 2没有Static Harness行。本次没有运行MemP/MemRL，不能将它们当作Static。论文MemP平均65.16、MemRL64.75，均不是本次结果。

本次Stability平均比论文低14.46分；Plasticity低16.71分。VQA分别比论文低43.66和45.47分。跨底座、数据规模与协议的差值只作描述，不能当成同设置数值复现误差。论文HCL相对Zero-shot有明显平均增益，本次没有观察到这一结论。

## 2. 版本、环境与设置差异

- 官方代码固定提交：`a6e5a01536ac8d42ca28f0a82efa223869c8c48d`，来源及MIT许可见[UPSTREAM.md](UPSTREAM.md)。本仓库是有明确修改的核心代码快照，非官方仓库。
- 硬件：RTX PRO 5000 72GB；4臂在4张显存足够的卡并行；共享模型只读复用，没有复制权重或修改其他实验。
- Python3.12；Torch2.7.1+cu128；Transformers4.51.3；pycocotools2.0.11；pycocoevalcap1.2；Java21。依赖固定项见requirements-reproduction.txt，CUDA轮子需按硬件选择。
- 论文Qwen3.6-27B，本次Qwen2.5-VL-7B-Instruct BF16，eval、requires_grad=False、temperature0，不训练权重。所有角色使用同一底座实例，不使用付费API。
- 论文每任务250 adaptation/50 validation/500 test；本次50/10/100、batch25；seed20261022。精确论文sample IDs未确认。
- 顺序Detection→Caption→RefCOCO→VQAv2；每任务两批，每artifact最多3候选；4个artifact为task-interface、workflow、memory-selector、context。不启用外部工具。
- 每批从25 adaptation候选选10 anchor、每任务容量25，使原总anchor>=20/每历史任务>=15的门控可运行。此采样频次不同于论文/初始profile，不隐藏为原设置。
- 两个HCL除历史预算B=0与B=null(∞)外匹配；本次当前分数严格提升、格式门槛与逐anchor历史成功→失败限制未降低。不是用净精度抵消遗忘。

## 3. 数据、划分与隔离

完整COCO2014 train/val图像和检测/描述标注、RefCOCO-UNC、VQAv2 train/val问答下载完成，原包保留；一次必要数量检查123,287张图像。下载收据只证明工具完成和已知长度/数量，不声称做了全量哈希。

试验使用共享COCO val2014与RefCOCO train2014图像，四任务全局按image ID互斥划分train/validation/test；每任务每split每图最多一例，不同任务同split可共享图像。Detection询问该图首个排序类别，RefCOCO使用首句，存在采样偏置。全量数据下载不等于全量图像都参与推理。

新协议使用637张不同原图；新test全局排除旧pilot已经报告的test图像。公开的sample_ids_native.json与sample_ids_pilot.json只包含公共数据集样本ID、image ID、来源和划分，不含原始问答、图像或预测。

anchor由adaptation交互选择；专用anchor库仅Evaluator使用，不进入retrieval；候选提示只收历史汇总。manifest内validation的anchor别名在该模式不激活。test只作报告，原test forgetting probe关闭，每次test断言memory及组件版本不变。validation为候选评价，历史被选为anchor的adaptation交互可能同时是历史经验，此边界显式保留。

## 4. 原始失败、修复与恢复版本

1. 第一轮20/10/50四臂都退出0，但它是工程负结果：Qwen2.5-VL原生像素坐标与原0–1000请求冲突；question-only去重把不同图Caption anchor合并；每任务2 anchor（Caption实际1）不足以满足后续prompt优化门槛，后3阶段跳过。该轮不覆盖、不修改答案、不作为充分HCL验证。
2. 在adaptation上进行零新推理坐标诊断，并按[官方Qwen博客](https://qwenlm.github.io/blog/qwen2.5-vl/)确认像素接口。固定前4检测+前4定位真实图预检8/8格式有效，无test参与接口选择。可选NativePixelModel只在明确单bbox请求时将单元素bbox_2d列表规范为单bbox对象，不猜坐标、不挑多框、不看标签，原始输出保留。
3. anchor增加opt-in visible_input身份（question+image ID/path，不含答案）；默认旧模式不改，旧运行不重载。
4. 独立Java解包的配置软链接最初错误，造成早期caption fallback。修复后实际pycocoevalcap验证通过，两个旧baseline只对保存输出纯评分，原fallback保留。曾误用官方raw resume重发3次推理，立即停止、保留尝试和成本；之后没有用该入口评分。
5. 严格native_v1的两个HCL都在Caption遇到两次非法memory role而退出1，原始失败与不完整矩阵单列保留。各有3条未提交raw残留，先隔离，再恢复最新已提交checkpoint；没有按测试分数挑checkpoint。新增candidate异常事务回滚保护。
6. **另立memory_selector_fail_closed_v1恢复协议**：仅在原两次memory-selector schema尝试耗尽后，将整例记空答案、格式无效、得分0，不调用最终回答、不映射非法role、不额外重试、不从分母剔除。其他异常仍fatal。每个符合原因的失败臂仅恢复一次，从时间最近的已提交checkpoint续跑；全部原调用和重放成本保留。完整策略见protocols/recovery_protocol_v1.json。

| 严格native_v1 | 原始状态 | 恢复版状态 |
|---|---|---|
| Raw Zero-shot | EXIT0 | 复用未触发异常分支的原输出 |
| Static Harness | EXIT0 | 复用未触发异常分支的原输出 |
| Stability-HCL | Caption schema错误，EXIT1 | 独立目录恢复，EXIT0，四阶段完整 |
| Plasticity-HCL | Caption schema错误，EXIT1 | 独立目录恢复，EXIT0，四阶段完整 |

不能把恢复版完成写成原严格协议成功。协议和修复在测试结论前独立记录，没有为追正结果反复搜索。基础模型始终冻结。

## 5. 四阶段矩阵与遗忘

每个阶段测试所有已见任务，每任务100例，共1000个阶段测试槽；之后另有一次400例最终汇总test，含缓存复用，不能将1400槽说成1400独立样本。最终实际不同test样本为400条。

### stability_hcl

| 学完任务 | Detection | Caption | Grounding | VQAv2 |
|---|---:|---:|---:|---:|
| 1 | 59.6434 | — | — | — |
| 2 | 60.1626 | 38.0068 | — | — |
| 3 | 60.1626 | 38.0068 | 84.0000 | — |
| 4 | 60.1626 | 38.0068 | 84.0000 | 35.6667 |

FinalAverage=54.4590；AverageForgetting=0.0000个百分点；BWT=0.1731个百分点。

### plasticity_hcl

| 学完任务 | Detection | Caption | Grounding | VQAv2 |
|---|---:|---:|---:|---:|
| 1 | 61.2111 | — | — | — |
| 2 | 56.3758 | 36.4010 | — | — |
| 3 | 51.0714 | 36.5853 | 83.0000 | — |
| 4 | 51.0714 | 36.5853 | 83.0000 | 34.3333 |

FinalAverage=51.2475；AverageForgetting=3.3799个百分点；BWT=-3.3185个百分点。


AverageForgetting严格按附录C.2公式11：前三任务各自从学会时至最终阶段的历史最大分减最终分，再取平均。BWT是最终分减首次学完分，单独列出。B=0限制anchor损失，不保证test遗忘严格为0；本次观测0不应解释为普遍保证。

Detection主分为label-matched micro-F1@IoU0.5；另存标准COCO AP，不能把二者互换。Caption用官方代码Bleu4/METEOR/ROUGE-L/CIDEr综合主分，未启SPICE；Grounding为Acc@IoU0.5；VQA为标准答案归一化后的consensus分。Table2未详细展开每个主分构成，因此不声称评价细节已经与论文完全一致。

## 6. 门控、记忆和运行时检查

合并原严格前缀与恢复目录：Stability memory门控8次接受1次，prompt组件更新32次接受3次；Plasticity memory门控8次接受2次，prompt组件32次接受4次。拒绝原因与计数见aggregate_results.json；更新次数不等于候选数，每组件最多3候选。没有用test选择更新。

Stability最终raw10/abstract2/anchor80/skills0；Plasticity raw20/abstract4/anchor80/skills2。Plasticity最终阶段保存的400条预测中43条选择了非空skills；只说明实际选用，不提供技能因果增益证据。Stability该数为0。

恢复目录中已知schema失败事件Stability11、Plasticity17；事件跨adaptation、候选和重复test，不是独立样本数。Plasticity第四阶段VQA测试2例失败，最终汇总test又记录相同任务下2次失败，仍保留100例分母；Stability没有报告test阶段的此类事件。每臂13条test只读收据全部通过，四阶段各任务total均100，参数冻结断言通过。原严格阶段只读收据另保留，不拿恢复检查替代原失败。

原实验源码及最终公开快照均在服务器复核：29项离线测试全部通过无skip（公开快照测试收据results/public_release_tests.txt），涵盖逐anchor成功→失败不净抵消、拒绝memory回滚、异常回滚、test无学习写入、native schema、anchor可见身份、失败分母与非目标异常传播。公开快照本机复核29项，27通过/2项因COCO依赖缺失skip；服务器完整29项是实验环境证据，不把本机skip写成通过。

## 7. 成本与预算

| 阶段 | 方法 | 模型调用数 | 累计模型调用秒数 | 峰值显存GB（十进制） |
|---|---|---:|---:|---:|
| local_20_10_50 | raw_zero_shot | 200 | 162.9 | 16.85 |
| local_20_10_50 | static_harness | 558 | 1637.7 | 17.09 |
| local_20_10_50 | stability_hcl | 1481 | 4742.7 | 18.36 |
| local_20_10_50 | plasticity_hcl | 1481 | 5213.1 | 18.36 |
| native_pixels_50_10_100 | raw_zero_shot | 400 | 490.7 | 16.85 |
| native_pixels_50_10_100 | static_harness | 1104 | 2885.4 | 17.08 |
| native_pixels_50_10_100 | stability_hcl | 949 | 2430.5 | 20.57 |
| native_pixels_50_10_100 | plasticity_hcl | 1706 | 4567.3 | 20.69 |
| native_fail_closed_v1 | stability_hcl | 6534 | 14441.1 | 18.70 |
| native_fail_closed_v1 | plasticity_hcl | 5194 | 10311.7 | 18.71 |

以上逐次generate记账合计19,607次。另有初始真实图smoke 1次、native预检8次、误评分额外3次，**已知总计19,619次**。误评分3次及预检未纳入上述逐次计时表，不能声称所有调用耗时完全覆盖。没有可靠token计账，不编造token成本或按缓存命中算新模型调用。

正式native对照含原失败和恢复：Raw400、Static1104、Stability949+6534=7483、Plasticity1706+5194=6900次，共15,887次。缓存复用减少实际调用，原失败重放也如实加计。category/phase/scope汇总见JSON；部分调用缺task_id，按任务ID推断的current/historical类别只是日志启发式，phase总数更可靠。旧pilot的scope wrapper没有可靠区分所有router调用，故不伪造精确旧分项。

Raw运行约8.43分钟、Static48.39分钟；Stability恢复worker5.04小时、Plasticity恢复worker3.74小时，恢复时长不含原失败前缀。并行调用秒数不能相加当wall time。首次共同预算到最后监督退出约9.74小时，所有下载、预检、修复、失败与恢复共享同一起点，未重置24小时预算，23.5小时提前终止守卫未触发。

最终保守新增总占用42.636GB；其中数据约40.946GB、非数据1.690GB（含首次共享pip/tmp新增的1GB保守预留）。非数据上限40GB/提前39GB，原总量上限严格50GB/提前47GB，均满足。既有只读模型与历史缓存不重复计入，也未删除历史缓存。运行实际峰值显存约20.69GB，来自原严格Plasticity；只报恢复峰值18.71GB会漏掉失败成本。

旧守卫曾因解压重命名导致du遇ENOENT退出，原实验未重启；随后原位接管恢复硬守卫。新守卫只容忍确为并发文件消失的du错误，其他测量失败fail-closed。所有推理进程已自然结束，不再扩大样本、加seed、换模型或搜索新路线。

## 8. 发布内容与复用方式

本仓库发布核心源代码、官方版本差异patch、匿名化配置模板、原协议、公开数据集sample IDs、聚合指标/门控/成本、离线测试摘要与本报告。未发布原数据/权重/逐例问答与预测日志/密钥/个人服务器路径配置/含逐例内容checkpoint/第三方PDF。

复现入口见README.md。公开r000/r001/r002为路径参数化的打包入口；r001保留原审计行为。r002是新整理的fresh-run监管器，未实际再启动GPU实验，不能把它称为已做过第二轮端到端复测。fresh fail-closed replay从头运行，与本次先strict失败再恢复的历史轨迹不完全相同；历史报告结果由公开ID与聚合收据保留，需逐字节日志审计时在原环境受控检查，不能从公开汇总还原全部逐例输出。
