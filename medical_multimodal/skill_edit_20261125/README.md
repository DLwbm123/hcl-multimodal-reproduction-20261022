# HCL 可执行技能的有限编辑试验

在冻结的 Qwen2.5-VL-7B、SAM 和原有 HCL LangGraph 上比较两种候选修订方式。设计采用 [SkillOpt](https://arxiv.org/abs/2605.23904) 的轨迹驱动和有限编辑思路。此实现直接编辑 HCL 的 JSON 技能，并沿用 HCL 当前任务与历史任务的验收规则。没有运行上游的 Markdown trainer、慢更新或元技能优化。

## 接口与运行

- `proposal.py` 的 `propose(generate, task, evidence, draft, mode)` 接受一个生成回调、训练反馈及同任务草稿。返回经原 HCL DSL 校验的单个技能，修复后仍失败则返回 `None`。
- `mode="full"` 生成一个完整技能。 `mode="edit"` 每轮最多改两处，包括插入、替换或删除步骤，以及替换说明、触发条件描述或输出。空草稿仅允许一次创建。所有编辑都在副本上应用，整组通过校验后才交给执行器。
- `run.py` 将模型提案接入原 `SkillExecutor`、`GraphAgent`、`score` 和 `accept`。先在训练图上直接执行候选，再在开发集上观察控制器是否主动选择它。只有后者实际调用新技能，并满足质量或效率及历史保持门槛才能提交。
- 两组每个任务从同一份模型生成的草稿开始，每组一个修订名额，最多一次格式修复。开发集逐例反馈不返回给提案器。相同任务和技能库的图推理结果在本轮内复用。

用本地 Python 在此目录运行 `python3 check.py` 可检查编辑器。基线默认在同级 `skill_pilot_20261116`，也可通过 `PILOT_ROOT` 指定。实模运行需要完整私有试验目录，包括原先的模型环境、冻结协议、训练状态和开发数据。将本目录源码放入新的 `RUN_ROOT` 后，在 GPU 主机提供这两个环境变量并运行 `python3 launch.py`。启动器复用原实验的资源监督器，要求独立输出目录和中性进程入口。数据、模型、逐例轨迹均保存在私有环境。

## 评测范围

每个任务的候选提案使用历史训练集中前四条带评分轨迹，第五张训练图用于执行检查。开发验收使用历史验证集按原顺序的前四个来源组，两任务复用同组。两组均不注入额外原始经验。四组开发数据只支持接口和小样本策略诊断。详细比较口径与资源上限见 `design.json`。

## 上游补丁接口检查

`upstream_probe/run.py` 单独调用微软原版 `apply_patch_with_report`。它从四条 T1 训练轨迹提取工具序列和分数，让冻结 Qwen 给预设的单工具初始技能提出一条文本修改。补丁应用到 HCL 的 `localize.instruction`，在第五张训练图上分别检查直接执行和控制器自主调用。该检查最多 24 次模型调用，单独计时，不参与两组开发比较。它只检验上游补丁与 HCL 的互操作性。

独立运行时，把 `upstream_probe` 中的 `run.py` 和 `design.json` 放在新的输出目录，复制父目录的 `launch.py`、`check.py`、`proposal.py`，并复制随附的 `upstream/` 子目录，内含[固定版本的原版编辑模块](https://github.com/microsoft/SkillOpt/blob/dc0b5e654cafa1073a2819086151c1e0d13012f0/skillopt/optimizer/skill.py)和 MIT 许可证。其余启动方式与主试验相同。

首次官方接口检查收到模型输出的单条 JSON Patch。修订版明确把指令定义为只有一个 `content` 槽的文档，并允许将该槽上的单条 replace 转换为 SkillOpt 原生补丁。使用 `upstream_probe/design_v2.json` 作为新运行的 `design.json`，并同时提供 `patch_format.py`。旧运行保留原协议与结果。

## 普通工具调用入口

`tool_guidance/guidance.py` 的 `GuidedEpisodeTools` 继承原执行器的工具集合，在实例化时接受冻结的 `guidance`。它只在控制器自主选择 `localize` 时附加该指令。已有 ROI 和控制器指令保留，组合文本超出原限制便跳过注入。`tool_guidance/run.py` 直接向原 GraphAgent 传入这个工具对象。它用相同初始/更新指令对四组历史开发样本作配对比较；两个条件均挂载该适配器，均不给控制器注册任何额外技能。

此检查使用独立输出目录：复制主试验 `launch.py`、`check.py`、`proposal.py`，以及 `upstream_probe` 的 `patch_format.py` 和 `upstream/`。将 `tool_guidance` 下的三个源码/协议文件也放在该目录中。除原有环境变量外，`PATCH_RUN_ROOT` 指向成功的上游训练检查目录，读取冻结补丁的私有凭据。
