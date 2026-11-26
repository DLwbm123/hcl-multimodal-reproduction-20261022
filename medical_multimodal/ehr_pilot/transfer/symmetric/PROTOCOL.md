# 两种选择性记忆策略的历史配对复测

沿用上一轮冻结的 11 条示范、10 张卡、检索函数、同一模型和工具、同一 83/77/40 划分、五分片顺序、批次大小 2、贪心解码、八步上限及原取证评分。新增推理仅为 77 道验收题各执行一次完整示范。按题面检索到的工作流，分别比较完整示范和无记忆。改对至少 1 题且改错 0 题时启用，其他回退无记忆。保存示范名单后才合成历史策略；流程卡沿用既有冻结名单。历史结果已公开，此轮是固定面板的对称对照。

在 40 题上逐题复用对应输入的原始贪心执行，对比首次合格成功。主比较以选择性示范为参照，统计选择性卡的改对、改错、共同成功、共同失败、净增及成功率差。另统计各自保留无记忆原成功的题数、按工作流的差异及在线开销。按患者聚类重采样给出描述性区间。

在新目录只读复用历史分片，独立拷贝只读原始 FHIR 快照并启动本地服务，沿用原回环端口。模型和缓存就地使用。新增目录原占用为零，预计数据库及日志低于 8 GB，单进程文件大小限制、2 小时总硬截止。只公开汇总结果与代码。

## 复现

准备独立输出目录，沿用上轮 SOURCE、MODEL、PRIOR、PYTHON、PREPARED，把 FHIR_BASE 固定为 http://127.0.0.1:18478/fhir/ 。将上轮的 compare.py、pilot.py、diagnose.py 和 library.json、partition.json、gate.json、valid_plain_*.json、history_*.json 只读链接到该目录。将 run_validation.py、evaluate.py 分别放为 v.py、q.py，放入本轮 launch.sh。将 calibration/start-db.sh 的监听端口 18476 改为 18478，放作 s.sh。指定 GPUS 为五个可用 GPU 的逗号分隔列表，以总超时启动 launch.sh。随后设置 SHARDS=5 和 V1_RESULT 为原 result.json 路径，在同一工作目录依次运行下列命令。

```sh
"$PYTHON" q.py freeze
"$PYTHON" q.py evaluate
```

demo_gate.json 只读取验收轨迹。comparison.json 的每一行配对来自同一个历史病例。逐题执行轨迹随原始数据保存在服务器上，公开文件仅含聚合数。
