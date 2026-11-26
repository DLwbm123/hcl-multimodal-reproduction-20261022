# 复现入口

本目录保留检索实验的冻结输入 edit_9.json、程序、协议和聚合统计。将 eICU Demo 2.0.1 的 patient.csv.gz、lab.csv.gz 分别链接为私有运行目录的 p.gz、l.gz。模型为 Qwen2.5-VL-7B-Instruct。SOURCE 下的 artifacts/source/c 应指向原始 MedAgentBench 代码及 data/medagentbench；MODEL 指向本地模型。PRIOR 指向既有开发目录；共享入口在导入时读取该目录的 split.json，本实验不使用该历史划分。如仅复现本实验，独立空目录中放一个内容为 {} 的 split.json 即可。建立仅本机可达、无既有患者的 HAPI FHIR 实例，服务地址为 http://127.0.0.1:18478/fhir/，导出 FHIR_BASE 为此地址。

在私有运行目录将 prepare.py、run.py、../../external/import_records.py 分别链接为 b.py、w.py、i.py，另链接 edit_9.json。将 ../../transfer 加入 PYTHONPATH，确保可以导入 compare 及其关联模块。调用 python b.py、python i.py、python w.py control。参考路径全部通过后，设置 PART 为 0、1、2 分别在三个设备上运行 python w.py infer。三个分片结束后调用 python w.py report 和 python audit.py。launch.sh 记录实际后台运行的工作顺序，依赖主机私有 env 和 db.pid。cases、resources、保留 ID 与逐例轨迹保留在私有运行目录。prepare.py 默认沿用本轮的种子、人群和标识起点。
