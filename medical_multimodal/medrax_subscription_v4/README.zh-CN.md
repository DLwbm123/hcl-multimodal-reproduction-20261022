# Sol High 订阅控制器：作者六工具医学 Agent 基础验收

按用户要求，用本地 ChatGPT 订阅的 **gpt-6.1-sol / high** 替换作者 GPT-4o，先完成不带 HCL 的 MedRAX 基础验收，再考虑持续学习。该目录不改变此前 MedRAX/ReAct 失败记录。

已完成 CLI 修复及真实订阅测试：纯结构化回复、图像识别后的工具选择、匹配工具结果的消费均通过；后两项证据见 [TRANSPORT_RECEIPT.json](TRANSPORT_RECEIPT.json)。原生 Agent 的工具调用 ID、ToolMessage 回传、第二轮 MemorySaver 状态与冻结面板图片的 CPU 检查通过。

作者 `dae30e2f136ef0b2a40885a4c335386e9ffad052` 的 `experiments/benchmark_medrax.ipynb` 使用六个工具，按原顺序为报告生成、分类、分割、MAIRA-2 定位、CheXagent、LLaVA-Med。MAIRA-2 和 LLaVA-Med 使用作者的 8-bit 加载配置。保留原系统提示、问题提示、LangGraph 工作流及第二轮只答 A–F 的评分流程。

必要差异是：Sol High 替换 GPT-4o；订阅 CLI 不提供原 `temperature=0.2/top_p=0.95` 控制；工具协议经结构化 JSON 转为原生 AIMessage，平台仍保留 CLI 封装。关闭宿主工具、网页、技能和记忆，并以本地沙箱隔离标签路径。图像通过本地附件传输，在原分辨率转换为 RGB PNG；原数据保留。不能将这些配置称为作者同模型数值复现。

## 冻结范围与状态

从未用于此前 48 病例实验的病例中，按 seed 20261026 和元数据类别选择 12 个不同病例，共 19 张图像；成员清单私有且已在推理前冻结。每例新建会话，不跨病例学习，不引入 HCL。全部病例保留分母，答案和解释不进入模型或工具，整批结束后才评分。

MAIRA-2 官方授权已取得并实测可读。两个新增模型合计约 42.65 GB，当前后台下载；原四工具权重、CLIP、数据与环境只读复用。下载完成后后台程序自动检查 GPU 余量、加载六工具、逐工具预检并运行冻结面板。**当前尚未完成六工具加载或医学任务验收。**

预算从首次订阅测试的原始起点统一计算，23 小时硬截止、总订阅调用上限 200、每台服务器新增严格小于 50 GB，45 GB 停止写入。下载使用有界 Range 续传与独立守护；初始完整 HTTP 响应截断已记录并保留部分文件及原时钟。短期下载地址定期从本机刷新，账号凭据不上传服务器。工程预检通过不等于医学准确性或持续学习有效性。

## 代码与记录

- [protocol.json](protocol.json)：冻结选择、作者对齐、差异、预算与验收边界。
- [run_agent.py](run_agent.py)：原生六工具执行、图像传输、独立病例状态、预算与最终评分；复用既有 `medrax_v1/native_runtime.py` 的兼容适配，部署时放入同一 code 目录。
- [subscription.py](subscription.py) / [rpc.py](rpc.py)：本地订阅及远程队列；认证仅留本机，真实医学工具在服务器执行。
- [download_assets.py](download_assets.py)：官方授权短期 URL 的分块下载、原时钟与存储守护。
- [check_contract.py](check_contract.py)：原生图调用/结果匹配及两轮状态的 CPU 检查。

公开仅源码、协议与无敏感内容的汇总回执。认证、签名下载链接、问题/答案、图像、病例成员、逐例结果和模型权重均不公开。沿用本会话唯一 30 分钟监测，完成后补充全部结果及负证据。
