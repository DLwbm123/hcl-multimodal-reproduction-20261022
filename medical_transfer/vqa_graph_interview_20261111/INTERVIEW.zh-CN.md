# VQA 学习状态接入 LangGraph：面试讲法

2026-11-11 已完成真实接入验证。原来的 VQA 74/200 结果使用直接视觉语言模型；这里把该实验实际学到的记忆载入，再经 LangGraph 的 Agent 循环完成回答。本次是执行流程与异常恢复验证，没有重新测量200例效果，也没有再次训练或选择学习状态。

## 可以怎么介绍

> 我用 langchain-core 的消息类型统一模型动作和工具结果，用 LangGraph 显式实现有状态的 Agent 循环。控制器决定调用哪个工具，工具结果通过 ToolMessage 返回，控制器再决定继续还是结束。外层的 HCL 负责经验检索、反馈学习、候选验收和回滚。图中的执行状态与跨任务学习记忆是两套不同的状态。

## 实际代码在哪里

| 文件 | 阅读点 |
|---|---|
| `../graph_integration/graph_agent.py` | `AgentState`、`process()`、`execute()`、条件边、`InMemorySaver`、`graph.invoke()` |
| `code/vqa_graph_bridge.py` | `attach_graph()`：只更换最终回答模型，保留已有选择/评审/记忆生成后端与学到的状态 |
| `code/worker.py` | 先恢复已学习状态，再接图，调用原 `pipeline._predict()` 并验证只读状态 |
| `code/check_graph.py` | 消息闭环、标签隔离、有界纠错、重复调用终止的脚本模型检查 |
| `../../hcl/pipeline.py` | `_predict()`、`_evaluate_train_batch()`、`_train_batch_and_optimize()`：外层反馈及更新 |

这几个文件是实际本地文件名，不要沿用旧仓库示例里的 `run.py` 名称。运行入口仍依赖已有授权数据、模型和旧状态，通过运行目录中的配置提供，不把这些私有资源放进代码包。

## 一次任务怎么走

```text
HCL 整理输入、选择历史经验、构建上下文
    ↓
GraphAgentModel.generate()
    ↓
process：模型生成动作
    ├─ 合法工具调用 → execute → ToolMessage → process
    ├─ 参数不合法 → 记录错误 → process 再生成一次完整动作
    └─ 最终回答 → END
    ↓
返回 HCL；训练时反馈才进入外层学习，评估时禁止更新
```

当前 VQA 接口：每题最多2次工具执行，最多1次动作格式纠错，因而最多4次控制器生成。默认旧接口的纠错次数仍为0，避免改变既有实验含义。超过预算、纠错后仍非法或越权工具名都不能变成成功答案。全局模型调用和运行时间也有独立预算。

## 本次真实运行观察

- 固定使用此前已用训练数据的前3例，不从新确认集挑演示案例。
- 首轮完成2/3例；1例在第二次请求工具时漏掉`instruction`，严格校验使它失败。
- 增加一次有界纠错后，原3例均完成。失败案例的实际事件顺序是：
  `process → execute → process → validation_error → process → execute → process`。
- 其他两例均是：`process → execute → process → execute → process`。
- 控制器动作由真实模型生成，没有使用预设工具覆盖计划。虽然注册了6个工具，实际这3例都选择了`answer_visual_question`，每例调用2次；不能声称实际演示了6种工具或证明模型选工具最优。
- 三例分别检索到2、2、1条历史记忆。原学习状态的10条raw、2条abstract、72个锚点和0条skill全部保留，读取前后快照相同，模型参数冻结。
- 两次尝试共47次模型调用（首轮22，纠错版25），从原始启动计时约254秒，进程正常退出。3项新增CPU检查与7项既有回归检查通过；CPU检查用脚本回答，不作为医学效果证据。

## 面试常问的区别

**LangChain与LangGraph有什么区别？**
LangChain提供模型、消息、工具和Agent等较高层抽象；LangGraph侧重显式的状态与控制流编排。我的项目实际使用`langchain-core`的消息对象和LangGraph的图API。没有使用`langchain.agents.create_agent()`，也没有实现一个通用RAG系统。

**你具体用了哪些消息对象？**
`HumanMessage`表示输入或控制器校验反馈，`AIMessage.tool_calls`表示已经通过项目校验的工具动作，`ToolMessage`携带工具输出。`tool_call_id`对应请求和响应，不能把工具文本随意当成另一条用户指令。

**模型是否原生支持function calling？**
这里本地Transformers后端输出JSON动作，由`decode_action()`检查名称、参数和格式，再转成`AIMessage.tool_calls`。这是项目适配，不是调用`bind_tools()`得到的原生工具调用，也没有使用`ToolNode`。

**StateGraph为什么需要state和reducer？**
状态包含消息、控制器次数、工具次数和校验错误。消息使用追加合并，计数和最近错误使用覆盖更新。条件边根据最新状态选择工具节点、回到控制器纠错或结束。

**checkpoint与HCL记忆一样吗？**
不一样。这里`InMemorySaver`保存单个episode的图状态；每次生成创建新图和新thread ID，没有跨进程持久恢复保证。HCL记忆与checkpoint另行保存，用来跨任务延续经验和恢复已接受的学习状态。

**这是自进化吗？**
外层有反馈驱动的记忆和提示候选更新机制；内层图负责执行。本轮接图联调只读取之前学到的状态，没有新一轮自进化训练。图拓扑、工具代码和基础模型权重都没有自动改变。

**怎么防止死循环与工具错误？**
工具次数、控制器次数、格式纠错次数、图递归深度和整个进程的时间/调用预算都有界。漏参数时把错误返回控制器，让它重新给出合法动作；不会由程序猜填参数，也不会执行非法动作。

**为什么还要LangGraph？**
对这个项目，它让循环和异常分支显式可读，并留下可检查的状态轨迹；方便解释与定位控制器和工具边界。它本身不会自动提升医学准确率。

官方概念参考：
- [LangGraph API overview](https://reference.langchain.com/python/langgraph/overview)
- [LangChain API overview](https://reference.langchain.com/python/langchain)
