# MedicalHarness 受限工具 H₀（工程验收）

这是 MedicalHarness MedTool 的适配执行协议，复用官方 MH-Lab LangGraph 循环和确定性评分器，不是原版 shell/文件工具环境复现，不是操作系统沙箱。

上游固定版本：`REAL-Lab-NU/MedicalHarness@8727186f04c25646e7ee1b9428f3b6dcd4239382`。本目录不复制第三方实现；运行时从用户提供的检出导入。MH-Lab 的规划、历史和循环继续复用；v1 使用 planning=on、context_policy=none、tool_bridge=off。每例最多24轮、80个数据/计算/提交工具请求，单轮真实生成最多1024 tokens，模型上下文16384 tokens。

## 工具边界

- `read_facts(offset, limit)`：只返回宿主事先加载的当前任务事实，每页最多100条；保留错误患者、单位和标本干扰项，不替模型选行。模型不能提供路径、SQL或过滤器。
- `calculate(expression, variables)`：白名单 AST 数学运算；有限数值、表达式长度/深度/节点数/指数上限；不使用 eval/exec，不支持属性、索引、导入、循环或任意函数。这是数学工具，不是经过临床验证的公式库。
- `submit_answer(calculators)`：一次结构化提交；模型没有输出路径参数。姓名字段实际为计算器名称，不是患者姓名；每行包含 name、episode_id、value。
- 上游 `update_plan` 和按配置启用的 `recall_event` 仅访问本例内存。所有其他模型工具请求失败关闭，绝不回落到上游 shell/文件执行器。

宿主是可信 Python 程序，仍具有账号权限；此设计限制的是模型可调用的接口，不能防御宿主/依赖漏洞，也不能宣称系统级隔离。真实任务的所有文件读取和结果保存必须由可信宿主预先指定，oracle不得传入工具或模型。`RestrictedTools.from_fixture` 只接受宿主提供的固定 SQLite 文件并以只读方式复制规定列；模型只接触内存快照。

## 模型与学习边界

当前冻结 Qwen2.5-VL-7B-Instruct BF16、贪心解码，不训练权重。checkpoint 的 chat template 未定义原生 function-call schema；适配器使用其多轮消息模板加显式 JSON 契约。无效 JSON 返回失败观察，不修造模型答案。调用和时间上限包括修正尝试。

后续 HCL 可以优化程序性记忆、工作流/上下文/工具选择指令；工具实现、白名单、评分、输出契约、权重和数据角色冻结。Static 与 HCL 必须复用同一个执行器。当前没有接入优化器，也没有开始跨病例学习。

## 检查与复用

先安装固定版本 MedicalHarness 的依赖。边界测试无需模型：

```bash
python -m unittest discover -s medical_h0/restricted_medtool -v
python medical_h0/restricted_medtool/acceptance.py --mode scripted \
  --output /your/new/synthetic-output --medicalharness /your/MedicalHarness
```

`acceptance.py` 中只有两个人工合成样例：BMI算术和两步算术链；故意包含患者/标本/单位干扰。脚本式通过只说明接线，真实模型通过也只说明这些工程样例，不是医疗准确率或持续学习效益。32个 MedTool 和107个 MedCTA 外部评估任务均未用于本次验收。

真实模式还需本地 HCL模型加载器、checkpoint、独立预算守卫和新输出目录中的 `budget.json`/`guard_status.json`。它们必须记录共享原始时钟、累计调用、存储阈值、工作进程身份。真实模式在每次调用前检查守卫；没有守卫不启动。私有服务器启动包装器不在公共源码中，不能将此命令当作无约束后台启动器。

验收失败必须保留原结果。正式 H₀验收还需要未用于修正的新样例、更多计算/缺失数据/单位转换覆盖及独立医学任务证据；这些门槛未达到时不启动HCL持续学习。
