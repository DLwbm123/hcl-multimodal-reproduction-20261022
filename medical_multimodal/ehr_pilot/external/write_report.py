"""Render shareable aggregates; the input contains no patient records."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
r=json.loads((root/'result.json').read_text())
n=r['overall']; lab=r['lab']
names=['姓名与生日查询','年龄','血压记录','最近镁值','条件补镁','平均指尖血糖','最近指尖血糖','转诊','条件补钾及复查','糖化血红蛋白及条件复查']
lines=['# 外部队列的冻结策略配对验证','',
       f"100 名 MIMIC-IV Demo 患者各执行 10 道任务。无记忆、选择性完整示范、选择性流程卡首次取证合格数分别为 {n['plain']}、{n['demo']}、{n['card']}，分母为 1,000。",
       f"从示范切换至流程卡，改对 {n['gain']} 题，改错 {n['loss']} 题，净增 {n['gain']-n['loss']} 题，共同成功 {n['both']} 题。两组分别保住原无记忆已成功的 {n['demo_retained']}/{n['plain']} 与 {n['card_retained']}/{n['plain']} 题。",
       f"按 100 名患者整组重采样 10,000 次，成功率差的描述性 95% 区间为 {r['rate_diff_ci95'][0]*100:.1f} 至 {r['rate_diff_ci95'][1]*100:.1f} 个百分点。",
       '',f"六类使用真实检验记录的 600 题中，三组分别成功 {lab['plain']}、{lab['demo']}、{lab['card']} 题。卡相对示范改对 {lab['gain']} 题，改错 {lab['loss']} 题。其余 400 题使用模拟身份字段或固定写入模板。该数据来自新机构；每类只用一个工作流模板。",
       '', '| 工作流 | 题数 | 无记忆 | 选择性示范 | 选择性卡 | 改对 | 改错 |','|---|---:|---:|---:|---:|---:|---:|']
for i,name in enumerate(names,1):
    t=r['by_workflow'][str(i)]
    lines.append('| '+name+' | '+' | '.join(str(t[k]) for k in ('n','plain','demo','card','gain','loss'))+' |')
lines+=['','## 条件工作流','', '| 工作流 / 分支 | 题数 | 无记忆 | 示范 | 卡 | 改对 | 改错 |','|---|---:|---:|---:|---:|---:|---:|']
labels={'missing':'无记录','no_write':'有值，无需写入','write':'有值，需写入','old_write':'有过期记录，需复查','recent_no_write':'近期有值，无需复查'}
for key,t in r['by_branch'].items():
    i,s=key.split('_',1)
    if int(i) not in (5,9,10): continue
    lines.append('| '+names[int(i)-1]+' / '+labels[s]+' | '+' | '.join(str(t[k]) for k in ('n','plain','demo','card','gain','loss'))+' |')
lines+=['', '## 在线成本与数据来源','']
for a,label in [('plain','无记忆'),('demo','选择性完整示范'),('card','选择性流程卡')]:
    c=r['costs'][a]
    lines.append(f"- {label}的模型决策数为 {c['steps']}，工具调用数为 {c['tools']}。")
lines+=['','数据来源为 Johnson 等发布的 [MIMIC-IV Clinical Database Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/)。题面和评分沿用 MedAgentBench 十种既有工作流。任务内 POST 为模拟执行；结果衡量按协议完成任务的能力。详见同目录 PROTOCOL.md。','']
(root/'REPORT.md').write_text('\n'.join(lines))
