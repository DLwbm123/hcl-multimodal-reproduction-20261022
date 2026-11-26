# 新开发来源的分支覆盖核查

数据源为 Johnson 等的 [PhysioNet eICU Collaborative Research Database Demo v2.0.1](https://physionet.org/content/eicu-crd-demo/2.0.1/)。仅取 patient.csv.gz 与 lab.csv.gz。未执行模型，无病例得分参与选择。每个 uniquepid 固定取有有效出科时点的最小 unit stay ID，锚定入科后 24 小时或更早出科时间。仅纳入当时已测量且已发布的数值，两处单位需一致或对钾等量等效；同时间同修订时间的矛盾记录标未决。镁用近 24 小时，钾用截至锚点的最近一次。分支表示冻结基准的模拟操作要求，与实际临床处方无关。

机器计数见 coverage.json。此核查只确认独立数据来源的可用性，无法作跨库个人身份联结。用于下一轮前应先冻结按患者分离的构建和验收、三种分支构成、评分及成本，不把覆盖数作为成功数。
