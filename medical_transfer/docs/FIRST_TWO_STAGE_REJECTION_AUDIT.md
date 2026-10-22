# 前两阶段更新拒绝原因：只读审计

2026-10-22。仅分析已冻结分类、息肉定位两个阶段的checkpoint和evaluator门控记录；未调用模型、重算替代坐标分数、修改代码/设置或干预进程。24个提示候选及2次记忆候选的聚合证据见 `artifacts/pilot_20261022_v1/first_two_stage_rejection_audit.json`，医学逐例数据保留服务器。

## 结论

前两阶段没有保留更新，主要是**当前任务验证收益没有提高或明显下降**；定位提示候选还受到**坐标边界合法率达不到100%**的独立阻挡，另有**4个非法prompt artifact**根本没有有效执行。记录中没有因B=0历史锚点遗忘而拒绝的候选。不能将全部拒绝归因于历史约束过严，也不能把所有定位问题归因于resize。

以下分数均是门控validation上的24例，**不是之前报告的50例test成绩**。

| 阶段/候选类别 | 个数 | 当前验证成功数：基线→候选 | 合规率 | 实际拒绝因素 |
|---|---:|---|---|---|
| 分类：联合记忆/摘要/技能更新 | 1 | 3/24→2/24 | 100%→100% | 主指标下降4.17个百分点、正确数减少1 |
| 分类：提示组件 | 12 | 每个均3/24→3/24 | 全部100% | 全部无严格收益；其中8个连输出行为也不变 |
| 定位：联合记忆/摘要/技能更新 | 1 | 10/24→2/24 | 62.5%→100% | 主指标下降33.33个百分点、正确数减少8；不是格式差导致拒绝 |
| 定位：可有效评估的提示组件 | 8 | 每个均10/24→10/24 | 每个62.5% | 全部无行为变化、无严格收益，且未达到100%格式门槛 |
| 定位：非法提示artifact | 4 | 未产生有效评估（记录0/0） | 不适用 | 3个未定义模板占位符、1个缺少记忆安全字段；不可当作0%定位准确率 |

### 分类12个提示候选

- task_interface_structuring的Candidate 1/2/3：3/24，不变，行为变化0。
- router_workflow的enhanced_rationale / memory_selection_optimization / tool_usage_expansion：均3/24；行为变化分别3/1/1，但没有增加正确样本数。
- memory_selector的memory_selector_prompt_1/2/3：3/24，不变，行为变化0。
- router_context的enhanced_reasoning / memory_optimization / skill_expansion：均3/24；行为变化分别2/0/0。

分类阶段尚无历史任务锚点。不存在“候选有收益，只因历史遗忘保护被挡住”的证据。等分候选不会保留，因为门槛要求主指标严格提升（min_primary_score_delta=1e-9），不接受仅换措辞/换错误答案而正确数不变。24例上提高1例就是约4.17个百分点。

分类记忆候选有2个原本成功样本变失败、1个失败变成功，净减少1个正确。这是**当前分类验证集**的变化，不是历史任务锚点遗忘；其正式拒绝原因是current_task_primary_score_delta_below_minimum与current_task_correct_gain_below_minimum。

### 定位8个可执行提示候选

- task_interface_structuring的Candidate 1/2/3；
- memory_selector的memory_selector_prompt_1/2/3；
- router_context的enhanced_skill_selection / enhanced_tool_usage。

八个均与基线相同：10/24成功、15/24坐标合法、9/24原图越界，行为变化0，历史分类锚点成功→失败计数0。逐份保存输出检查：24例均为合法单bbox JSON、四个有限数值、正向框；9个不合规输出的原因都是**越出原图边界**，没有发现JSON语法/字段/数值类型错误。这里的62.5%是validation合规率，与先前test的60%是不同样本。

因此，定位提示候选确实存在坐标合法性造成的硬阻挡：即使改进正确数，只要合规率仍低于100%，也不能通过当前规则。但本轮这8个候选同时没有任何收益；**不能推断只修复格式就会得到可接受更新**。本次没有缩放/clip后计算反事实准确率，也不能从越界现象证明所有错误都来自processor resize。

定位记忆候选则相反：合规率从15/24升到24/24，但成功数由10降到2，原成功→失败9、原失败→成功1。说明“更合规”不等于“定位更好”。历史分类锚点记录forget_count=0，正式拒绝原因是当前任务收益倒退，而非历史保护或格式门槛。

### 定位4个非法artifact

| 组件/候选 | 原始验证错误 |
|---|---|
| router_workflow / enhanced_rationale | prompt_not_renderable: KeyError: historical_anchor_metrics |
| router_workflow / optimized_skill_selection | prompt_not_renderable: KeyError: skill_selection_rationale |
| router_workflow / enhanced_memory_selection | prompt_not_renderable: KeyError: memory_selection_rationale |
| router_context / enhanced_reasoning | missing_memory_safety_fields |

前三个是模型生成的模板引用了该组件渲染环境不提供的占位符；第四个没有保留要求的记忆安全字段。它们有明确artifact_validation_errors，属于候选合法性失败。系统随后附带的零收益/格式不达标原因使用的是空评估占位结果，不能与8个真实评估候选混为一谈，也不能解释成4个候选模型都输出错框。

## 历史约束与解释边界

20个可有效评估的提示候选中，正收益候选0个；分类12个没有历史锚点，定位8个使用24个历史分类锚点且forget_count均为0。两次记忆门控也没有历史预算超限的拒绝原因。本结论针对这两个已完成阶段，不是“历史约束永远不影响HCL”。

记忆摘要日志的跨任务compact汇总会省略部分比较计数字段；本审计以gate记录的forget_count、正式拒绝原因及实际代码的success→failure计数规则解释，不将缺省behavior_comparable_count当作新的运行故障。

数据规模很小，提示候选多为轻微措辞变化，部分组件在稳定记忆为空时难以影响最终输出；这是与记录相符的机制解释，不是经新实验验证的因果结论。定位接口问题限制了方法效果的解释，但不能掩盖本轮没有观察到收益及非法候选的问题。

当前实验与科学门槛保持原样。没有修复这些候选模板、重生成候选、放宽遗忘/收益/格式门槛，也没有接入新坐标层或启动后继对照。新接口开发验收中的负结果仍保留。
