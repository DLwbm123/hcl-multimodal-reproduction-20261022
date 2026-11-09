# 三任务 HCL + LangGraph 融合实现

本目录把原有医学三任务 HCLPipeline 与 MedRAX 风格的工具执行循环接在一起。分类、息肉定位、医学 VQA 的任务顺序、数据角色和医学评分器沿用三任务实现；最终作答改为真实 LangGraph 的 `process → execute → process` 循环。

这是独立的新实现与新协议。它不覆盖 2026-10-22 三任务实验，也不改写 2026-11-09 原生 MedRAX 两阶段整合结果。首次运行用于工程接通，不证明医学效果或持续学习收益。实际完成情况见本目录 `REPORT.zh-CN.md`（运行完成后生成）。

## 先理解两层循环

```text
run.py
  ├─ 加载一个冻结的 Qwen2.5-VL-7B
  ├─ construct_pipeline()：保留原医学数据、评分器和 HCL 更新流程
  ├─ integrate()：把最终作答接口换成 GraphAgentModel
  └─ HCLPipeline.continual_train()
       ├─ 分类：train → 候选验收 → test 分类
       ├─ 定位：train → 候选验收 → test 分类、定位
       └─ VQA：train → 候选验收 → test 分类、定位、VQA

每次 _predict_one()
  → TaskInterface.build_taskinterfacechunk()
  → Router.run()：工作流指导、经验选择、上下文组织
  → GraphAgentModel.generate()
       process：模型生成 JSON 动作
          ├─ tool → execute：调用当前图像的任务工具
          │                    → ToolMessage → process
          └─ answer → END
  → 医学评分器；只有学习交互进入可学习记忆
```

外层学习的是经验与提示组件；内层执行的是当前图像任务。LangGraph 的图拓扑和工具实现固定，不在本轮自动改写。HCL 的工作流提示可以变化，但它不是 LangGraph 的节点和边。

## 阅读顺序

1. `run.py/main()`：任务、模型、数据及整合入口。
2. `graph_agent.py/AgentState`：消息列表通过 reducer 追加；计数器覆盖更新。
3. `graph_agent.py/GraphAgentModel.generate()`：`process`、`execute`、条件边、`END` 和 episode checkpoint。
4. `run.py/integrate()`：为什么只替换 `pipeline.model`，而选择、记忆和候选生成模型仍使用底层生成接口，避免辅助调用也误入 Agent 循环。
5. `../../hcl/pipeline.py` 对应仓库根目录的 `hcl/pipeline.py`：阅读 `continual_train()`、`_train_batch_and_optimize()`、`_predict_one()`。
6. `../src/medical_hcl/bridge.py`：医学评分、训练/验证角色检查、测试只读检查。
7. `test_integration.py`：理解非法工具、循环耗尽、标签隔离及完整三阶段流程的验证方式。

## 这次合并实际保留和增加了什么

| 内容 | 实现 |
|---|---|
| 三任务顺序 | 原 `HCLPipeline.continual_train()`，不是重写三阶段脚本 |
| 学习对象 | 记忆候选；输入结构化、工作流、记忆选择、上下文四类提示候选 |
| 验收 | 原当前收益与历史锚点约束，未通过就恢复/保留旧状态 |
| Agent 编排 | 实际 `StateGraph`、条件边、`AIMessage.tool_calls`、`ToolMessage` |
| 工具选择 | Qwen 根据可见问题和 HCL 上下文生成 JSON；Python 校验后转为 tool call |
| 图像访问 | 工具仅获得运行时绑定的当前图像；模型不能指定任意路径 |
| Trace 到优化 | 当前训练批次的工具计数、完成数和错误摘要进入候选生成指标；不是因果归因引擎 |
| 执行状态 | 每次 episode 独立的内存 checkpoint；不承诺进程崩溃后持久恢复 |
| 学习状态 | 原 HCL 经验库、组件版本与持久 checkpoint；与图消息历史分开 |

三个工具 `classify_image`、`localize_region`、`answer_visual_question` 共享同一个冻结 Qwen，通过不同任务契约提供接口。它们不是三套独立医学专家模型，也不是 MedRAX 的胸片六工具。原胸片工具的任务适用范围不能自动扩展到息肉图像。

控制器使用结构化 JSON 适配，不声称 Qwen 的本地 Transformers 接口原生提供了 OpenAI 式 function calling。工具动作采用 `{"tool": ..., "instruction": ...}`；获得工具观察后，最终答案允许直接使用任务本身的格式，例如分类标签或 bbox JSON。非法工具请求仍然失败，不能当作普通答案。学习这个适配有助于区分“模型生成动作”与“框架执行动作”。

第一轮 v1 在控制器层又要求最终答案套 `answer` 字符串，和定位任务的 bbox JSON 产生冲突。v2 只修正此输出契约及控制指令边界，先用相同的三个训练样本检查图执行，再允许一次完整三阶段复测。两轮共用原始时间和模型调用预算，第一轮错误全部保留，评分和接受门槛没有改变。

## 首轮冻结范围

每任务 H-search、validation、anchor、test 各 1 例，选取规则固定为原 StrictMedicalLoader 的首个样本，不按新模型结果挑选。每个提示组件 1 个候选。历史锚点最小总数和每任务最小数预先设为 1，仅用于这个微型工程协议；旧三任务实验的 20/15 门槛没有改变。收益必须严格增加、历史成功转失败预算为 0、当前输出格式要求 100%。

首次作答前必须调用一个工具，这是明确的工程覆盖约束；工具名称和请求由模型生成。每个 episode 最多两次工具调用和三次控制器调用。不能据此宣称已学会“何时不调用工具”的策略。

最多累计 1200 次底层模型调用，首次原始时钟起 2 小时硬截止，独立 supervisor 监测新增文件并在 1 GB 前停止。模型、环境、原始图像与原划分只读复用。所有真实图像、问题、答案及逐例 trace 留在服务器私有目录；公开只含源码、协议与聚合报告。

## 运行方式

需要已有 HCL 源码、`medical_transfer/src`、本目录加入 `PYTHONPATH`，以及与现有模型兼容的 Torch/Transformers 和 LangGraph、langchain-core。路径通过环境提供：

- `RUN_ROOT`：新的独立运行目录，含只创建一次的 `budget.json`。
- `PREPARED_DATA`：已有冻结三任务准备目录。
- `MODEL_PATH`：本地冻结 Qwen 权重目录。
- `CUDA_VISIBLE_DEVICES`：经显存检查的设备。
- `WORKER_ENTRY`：中性文件名的入口，内部调用 `run.py`。

用中性入口运行 `supervisor.py`，由其拉起 worker；不要在已有 `STARTED.json` 的目录重复启动。预算与预检失败均保留原始时钟。

CPU 检查使用真实 LangGraph 和 HCLPipeline，但模型回答为脚本夹具，仅用于契约测试：`python -m unittest test_integration`。模型 smoke run 与 CPU 夹具结果分开记录。

## 面试表述

“我把系统分成内外两层：内部用 LangGraph 组织模型与工具的交互；外部用 HCLPipeline 按任务序列积累经验，生成并评测 Harness 更新。图执行状态用于完成一次任务，学习状态用于跨任务积累。模型权重冻结，候选通过当前收益和历史保持约束才保留。”

不要把它表述为已训练工具模型、已学得图结构、已解决遗忘或已证明医学收益。真实结果与失败应按运行报告回答。

LangGraph 概念参考：[官方 Graph API 文档](https://docs.langchain.com/oss/python/langgraph/graph-api)。
