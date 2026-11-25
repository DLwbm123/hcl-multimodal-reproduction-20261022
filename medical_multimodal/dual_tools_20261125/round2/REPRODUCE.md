# 复运行说明

两个实验共用同一冻结的 Qwen2.5-VL-7B-Instruct 和 HCL RawMemory、AbstractMemory 实现。新建空的本地运行目录作为 RUN_ROOT，并把 round2 的 Python 文件和所选 design_med.json 或 design_fundus.json 放到其中，将选定配置复制为 design.json。运行时保留如下项目模块在同一目录或 Python 路径下。

| 运行时模块 | 仓库内来源（以 medical_multimodal 为根目录） |
| --- | --- |
| shared.py、previous_learning.py、clinical.py、retina.py、tool_server.py | dual_tools_20261125 下的同名文件。MedCTA 后端从 RUN_ROOT 加载 tool_server.py |
| acquire.py | medthinkvqa_20261125/selection/run.py |
| patch_format.py | skill_edit_20261125/upstream_probe/patch_format.py |
| upstream_skill.py | skill_edit_20261125/upstream_probe/upstream/skill.py，保留该目录的 MIT 许可证 |

将 hcl.memory 所在包加入 PYTHONPATH。MODEL_ROOT 指向已准备的本地模型。使用与模型兼容的 PyTorch、Transformers，以及项目现有 Pillow、OpenCV、NumPy 和 scikit-learn 环境。每个实验使用独立目录，冻结候选后自动进行验收和留出评估。

- MedCTA 使用 STUDY=med，DATA_ROOT 指向含 tasks.jsonl 和对应图像的推理目录，CTA_ROOT 指向含参考轨迹、AgentLego 代码、state/runtime_environment.json 和 plan.json 的已就绪四工具环境。每个环境使用自己的路径配置。先运行 python check.py，再运行 python run.py。
- 眼底使用 STUDY=fundus，FUNDUS_ROOT 指向 REFUGE2 镜像解压目录，含 train/{images,mask} 和 val/{images,mask}。复用的加载器导入时也读取 DATA_ROOT，可将其设为 FUNDUS_ROOT。先运行 python check.py，再运行 python run.py。
- 两个实验都会写出 result.json。diagnostics.py 接收 fundus 或 clinical 及相应的 private_阶段名.json，生成事后分析汇总，仅在本地处理逐例记录。

本轮用了 HCL 原始与抽象记忆，以及 SkillOpt 的原生追加编辑接口。MedCTA 额外检索同阶段的训练决策，眼底从四个预定的数值决策规则选择一个。归因到任何单个模块均需另外做消融。运行需要已获得许可的数据与工具环境。公开材料仅含代码、协议和聚合指标。
