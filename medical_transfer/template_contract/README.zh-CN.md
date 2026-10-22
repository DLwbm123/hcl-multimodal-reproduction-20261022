# 候选模板结构修复与开发验收

2026-10-22。本目录为独立、显式opt-in版本；当前 `pilot_20261022_v1` 不导入、不热更新。原始4个非法候选、拒绝记录和原分数全部保留。本次没有新模型推理、H学习或HCL/Static对照。

## 修复范围

1. 按真实调用者收紧变量契约：workflow只提供 `model_visible_input`、`selected_skills`；context提供5个已存在字段；memory selector提供4个字段；task interface只提供 `raw_visible_input`。不再用所有组件变量的并集掩盖组件局部非法变量。开发检查解析已固定上游三个Router调用者的 `prompt_values`，与清单逐一一致。
2. 对已观察到的 `historical_anchor_metrics`、`skill_selection_rationale`、`memory_selection_rationale`，确定性移除引用它们的独立输入段、标题和依赖这些不可用输入的指令段，追加明确的可用输入边界。**不提供历史锚点、答案或伪造的rationale，也不以空字典冒充实际数据。**其余候选措辞尽量保持。
3. router_context缺失字段名时，补回标准安全声明：`correct_feedback`是权威监督，`previous_attempt`可能错误且不能照抄；保留内部推理与final-only输出要求。原候选将字段名改写为自然语言的版本仍保留在original记录中。
4. 未知新变量、同时包含合法/非法变量的混合段、缺失必需输入、可见思维链要求等仍拒绝，不尝试猜测性修补或新增生成重试。
5. 给后续候选生成请求附加组件准确的变量白名单、必需字段、明确的安全字段要求。候选数不增加，不用替补候选补齐修复失败，保持原始ID/版本与原artifact，并写 `structural_repair` 记录。

这是一份**新的候选规范化协议**：删除不受支持的候选内容与补回安全规则属于可审计的语义约束，不是对原始失败的重新解释。后续若使用必须在新协议中披露，不能覆盖本轮记录或沿用原分数声称修复后的性能。

## 验收结果

| 验收项 | 结果 |
|---|---:|
| 已观察到的非法候选 | 4/4修复后通过结构校验 |
| 既有合法候选作为对照 | 20/20保持原artifact内容不变 |
| 开发数据 | classification / polyp_localization / visual_qa 各4例，仅可见validation清单 |
| 最终验收矩阵 | 24模板×12开发例，共288个组合，全部可渲染 |
| 实际Optimizer规范化路径 | 24/24通过，无新增候选 |
| 新建真实MedicalPipeline安装钩子 | 通过，NoInferenceModel调用0 |
| 单元/契约测试 | 本机、服务器各8项通过 |
| 模型推理、权重训练、H更新、新对照 | 均0 |
| private答案、test、retention_anchor记录读取 | 均0 |

workflow与context使用真实Router渲染/上下文组装调用，TaskInterface使用真实结构化prompt渲染；memory selector使用与已核对的真实调用者完全一致的变量映射进行PromptTemplate渲染。workflow需要JSON响应的部分使用固定CPU夹具，不是模型效果证据。开发图像只被上游正常读取以构建附件元信息；没有复制医学图像。本地仅保存代码、候选prompt和聚合验收，逐例开发IDs只保留服务器。

本次验收只证明模板结构和调用兼容性。**不证明修复后能提升准确率、通过遗忘/收益门控或解决坐标问题。**已有坐标开发语义验收失败仍有效，没有放宽、重评或覆盖。

## 代码与接入

- `template_contract.py`：变量契约、确定性修复、严格校验、ContractOptimizer和显式安装函数。
- `test_contract.py`：8项边界检查，含未知变量拒绝、跨组件变量拒绝、字面JSON括号、保留安全规则、幂等性、候选数量/生成调用数不增加、所有Optimizer配置不变和旧状态拒绝。
- `validate_development.py`：开发集CPU验收，固定选择规则与记录；不读取私人答案，不能调用真实生成器。
- `source_artifacts.json` / `repaired_artifacts.json`：原始24模板与修复/动作收据，不含医疗逐例数据。
- `development_acceptance.json` / `cpu_checks.json`：聚合验收。

安装形态（仅准备，不是实验启动脚本）：

```python
from template_contract import install_candidate_contract
# pipeline必须是全新独立storage中的新实例，尚未生成候选或学习。
install_candidate_contract(pipeline)
```

安装会拒绝已有checkpoint、非零学习版本/批次、已使用的Optimizer、非空磁盘历史和重复安装。它只替换新实例的Optimizer类，完整复制其全部初始化配置，不改模型、数据、候选数、B、收益/格式/历史约束、评测器、坐标接口或预算。未对全局上游类作monkey-patch，未创建能自动启动实验的入口。

CPU复核可用 `PYTHONPATH=<pinned-upstream> python -B -m unittest discover -s <this-dir> -p 'test_*.py'`。服务器路径仍须通过中性入口或stdin传递，不放入新运行进程的可见argv。未来实验只有获得明确启动授权后才另行冻结协议与资源预算，不能因本次结构验收通过自动开始。
