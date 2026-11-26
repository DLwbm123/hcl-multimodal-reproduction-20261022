"""Render a public report using only per-group counts and error labels."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
result=json.loads((root/'result.json').read_text())
rows=['# 开发侧条件流程局部修订','',
      '同一冻结模型用建设侧错误记录为每个目标工作流生成一次局部编辑。验收患者与建设患者隔离。每一行均为旧策略和候选在相同病例上的配对结果。','',
      '| 工作流 | 来源 | 分支 | 题数 | 旧策略 | 修订后 | 改对 | 改错 | 共同成功 |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|']
for key,name in [('5','条件补镁'),('9','条件补钾及复查')]:
    for src,label in [('real','真实开发'),('simulation','模拟开发')]:
        for branch,kind in [('write','需写入'),('no_write','无需写入')]:
            g=result[key]['groups'].get(src+'/'+branch)
            if g is None:
                rows.append(f'| {name} | {label} | {kind} | 0 | — | — | — | — | — |')
                continue
            rows.append(f'| {name} | {label} | {kind} | '+' | '.join(str(g[k]) for k in ('n','old','new','gain','loss','retained'))+' |')
rows+=['','## 采纳与限制','']
for key,name in [('5','镁'),('9','钾')]:
    status='通过' if result[key]['adopt'] else '未通过'
    rows.append(f'- {name}流程候选{status}预设的开发验收。')
rows+=['','真实补镁验收中仅两例需写入，真实补钾验收中没有需写入的病例。模拟结果单列。MIMIC-IV 外部队列未用于修订或验收。','',
       '逐来源、分支和首次动作偏差的建设侧汇总在同目录 result.json 中。判定基于可见工具轨迹；JSON 与参考结构有差异的首个字段只能作为待核实的故障线索。']
(root/'REPORT.md').write_text('\n'.join(rows)+'\n')
