# 运行方法

沿用 paired_audit 已使用的推理环境。源病例按 VQA-RAD 原始 JSON 的 image_case_url 与 image_name 分组，与 MedCTA 准备目录中的 107 个图像作感知近似排除后抽样；freeze.py 的输出文件应落在新建的私有 PANEL_ROOT，目录创建时将启动 Unix 秒存入 start_epoch。EXTERNAL_ROOT 指向含原始英文数据表及 images/ 的目录，DATA_ROOT 指向既有 MedCTA runtime。使用项目 GPU Python 执行 freeze.py。

每个分片设置 PART（0–5）、RUN_GPU、PILOT_ROOT、LIB_RUN（含 token 记录的 paired_audit 运行代码）、FEATURE_RUN（feature_decision 代码）、BOUNDED_RUN（冻结规则）、OUTCOME_FIT_CACHE（原始拟合经验）、CTA_ROOT、HCL_SOURCE 及 DATA_ROOT。MODEL_ROOT 指向原 Qwen2.5-VL-7B-Instruct。REFEREE1 和 REFEREE2 分别指向 Lingshu-7B 与 Qwen2.5-VL-3B-Instruct。将本目录的 run.py、check.py、design.json 放在 PANEL_ROOT 后，在 bash 中运行 launch.sh。分片内的推理与两个评分器依次独占同一个可见 GPU；评分器逐一处理每题的不同答案，决策只见问题、原参考及一份去重答案。

影像辅助复核分两个分片（PART 0、1）。以同一 launch.sh 运行，将目标目录改为 audit$PART，以 evidence.py 替换该目录内的 run.py，仍使用原 start_epoch。此步骤等待已持久化的主实验评分并核验退出状态，然后执行逐例来源隐藏的独立评阅。核查范围为任一评审发现分支事实得分不同、两评审判定不一致或固定十分之一抽查命中的病例。每题只对不同的“答案＋已送达的观察”各评一次。记录原始分支输出、两位自动评审及辅助复核标签后，用 summarize.py 对 PANEL_ROOT 生成公开汇总。所有新分片均复用同一硬截止时间。外部复现需自行取得原始数据、冻结模型及可读取的历史策略材料。

工具数、模型请求和 token 按每条策略单独部署核算。整图对照 D 从头生成，触发时 G 复用其确定性结果；独立部署 G 时应计入其模型和工具开销。工具内部 OCR/计算不计为额外的语言模型请求。推理延迟为逐例分支墙钟之和，未含模型装载和事后评审。辅助复核 A/B/C 分别代表证据支持/证据不足或冲突/不确定；附加错误 A/B/C 分别代表有额外错误/未发现明确额外错误/额外细节无法核实。辅助复核集中包含刻意选取的困难例，不宜把其比例解释为全人群发生率。

主比较另做一次分支顺序隐藏的语义敏感性核查。对每个触发项，将 S 和 V 的不同答案按字典序排列，再用种子 661＋该项顺序号打乱，向助手提供问题和参考而不提供分支身份。其判定保存为私有的顺序号到 0/1/U 字符串映射。用 trigger_semantic_check.py 对私有运行根目录汇总。这一附加检查仅作敏感性分析。
