# 三任务 Agent 的六工具扩展

保留分类、定位、VQA 三个任务接口，增加图像质量统计、局部观察和框提示前景分割。本目录实现工程演示，不重跑原三任务对比，也不继承原实验的性能结论。

## 六个工具

| 工具 | 实际执行 | 返回内容 |
|---|---|---|
| `classify_image` | 冻结 Qwen 查看原图并按指定类别判断 | 类别文本 |
| `localize_region` | 冻结 Qwen 查看原图并提出目标框 | 模型生成的位置回答；本轮不评分 |
| `answer_visual_question` | 冻结 Qwen 根据原图回答问题 | 问答文本 |
| `analyze_image_quality` | OpenCV/NumPy 计算灰度统计 | 平均亮度、低亮度/高亮度像素占比、Laplacian 方差 |
| `inspect_region` | Pillow 真正裁剪输入图，再调用同一冻结 Qwen | 裁剪图文件、原图区域、尺寸与局部观察 |
| `segmentation` | OpenCV GrabCut 在给定框内执行前景分割 | 原图尺寸二值 mask、叠加图、像素面积、前景外接框 |

分割是传统颜色/外观建模算法，不是训练好的息肉分割网络，不代表能识别病理类别。质量统计未经医学质量标注校准，Laplacian 方差随图像尺度与内容变化，不能直接当作诊断质量评分。裁剪没有增加原图信息，可能遗漏全图背景。

## 动作格式

原三工具模式保持原动作格式。在新的六工具模式中，所有工具统一使用 `tool + arguments`；分类、定位和问答的参数只有 `instruction`：`{"tool":"classify_image","arguments":{"instruction":"从题目类别中选择一个"}}`。

新增工具使用严格结构化参数，坐标是原图整数像素、左上闭右下开：

```json
{"tool":"analyze_image_quality","arguments":{}}
{"tool":"inspect_region","arguments":{"bbox":[20,30,220,230],"question":"描述这个区域中可见的结构"}}
{"tool":"segmentation","arguments":{"bbox":[20,30,220,230]}}
```

运行时绑定当前图像，模型不能传文件路径。程序检查参数与图像边界；无效框返回错误观察，不伪造分割结果。GrabCut 的矩形必须留下框外背景。处理最长边上限 768，mask 用最近邻恢复原图尺寸，并记录处理尺寸。空前景仍如实报告；不生成假 mask 掩盖失败。

## 执行与经验学习的边界

共享的 `graph_integration/graph_agent.py` 支持可选 `tool_runtime`。未传入时仍只有原来的三个工具，原协议和已有结果文件不变。在 HCL 驱动的**新协议**中设置 `enable_image_operations: true`，并合理配置 `max_tool_calls_per_episode`，即可让相同图执行器暴露六工具；不要直接修改或复跑旧冻结协议。

本次 `demo.py` 读取既有 `polyp_localization/h_search` 可见记录的第一条，不读取参考答案、验证或测试。按图像尺寸固定取中央 80% ROI。三次模型路由尝试在参数或顺序处失败后，停止调整路由提示，改为程序明确编排六步，真实执行四次视觉模型推理（分类、定位、局部观察、问答），再调用模型汇总观察。Trace 用 `decision_source=prescribed_plan` 标记程序动作，用 `model` 标记模型生成，不冒充自主工具选择。ROI 不是模型定位结果，也不是标注。

`segmentation` 的 mask 和面积等作为工具观察回传控制器；本轮控制器不另行查看叠加图。裁剪图实际进入局部 VLM 调用。质量与分割不调用 VLM，局部观察调用一次 VLM。

本次没有运行 HCL 更新与候选选择。它验证扩展的执行层能接通，不能证明新工具提高准确率、持续学习或保持性；也没有证明模型能自主稳定完成六工具选择。实际 Agent 模式不传 `coverage_plan`，仍由模型产生动作。旧三任务性能数据与旧三工具 63 回合验证分别保留。

## 运行与验证

依赖已有 Torch/Transformers、Pillow、NumPy、LangGraph/langchain-core；新增依赖 `opencv-python-headless==4.11.0.86`（仅安装在本轮隔离依赖目录，不改共享环境）。HCL 仓库根目录、`graph_integration` 和本目录应在 `PYTHONPATH`。

环境变量：`RUN_ROOT`、`PREPARED_ROOT`、`MODEL_PATH`、`CUDA_VISIBLE_DEVICES`。`RUN_ROOT` 需含 `protocol.json` 和一次性创建的 `budget.json`，后者含 `started_epoch`、`deadline_epoch`、`initial_bytes`、`stop_new_bytes`。通过既有独立 supervisor 和中性入口运行，最多 30 分钟、20 次模型调用、8 次工具调用、新增目录小于 1 GB。结束后停止，不继续实验。

`test_tools.py` 验证裁剪像素、标签隔离、真实 GrabCut mask、参数边界及六工具消息流程（模型为脚本夹具）。原有 `test_integration.py` 回归检查默认三工具和完整三阶段夹具流程。真实模型结果另见本目录报告。

## 面试表述

“我保留分类、定位和问答三个任务，在执行层增加图像质量统计、局部复查与框提示分割。控制器输出结构化动作，程序负责参数验证、实际图像处理与观察回传。分类、定位、问答和局部观察共享冻结 Qwen；质量统计和 GrabCut 是确定的图像处理工具。这次验证了一个规定顺序的六工具真实执行流程，尚未验证新增工具的性能收益。”

实现参考：[Pillow crop](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop)、[OpenCV GrabCut](https://docs.opencv.org/4.1.1/d7/d1b/group__imgproc__misc.html)。
