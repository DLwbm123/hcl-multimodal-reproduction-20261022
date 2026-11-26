"""Emit a local de-identified row audit or a publishable aggregate, after scoring."""
import sys,json
from collections import Counter
from pathlib import Path
from decision import moves,read

root,replay=map(Path,sys.argv[1:3])
rows=read(root,'private_test.json')
old=read(replay,'private_replay.json')
assert rows.keys()==old.keys()
selected=[(old[k],row) for k,row in rows.items() if row['disputed']]
abbr={'ImageDescription':'整图','RegionAttributeDescription':'局部','OCR':'OCR','Calculator':'计算'}
if len(sys.argv)>3 and sys.argv[3]=='table':
    print('# 23 道首步停止分歧题逐题回放\n')
    print('编号依照冻结评估顺序生成。基线前缀列从第一步开始，以 0/1 表示每一步重新作答的自动评分。旧策略和新版共享完整条款版的采信策略。\n')
    print('| 序号 | 基线第1→2个工具 | 首次采信 | 基线各步对错 | 旧版对错 | 新版实际调用数 | 新版采信第2步 | 新版对错 | 新版终止机制 |')
    print('|---:|---|---|---|---:|---:|---|---:|---|')
    reason={'router_stop':'基础路由器结束','stop_verified_quote':'引文验证通过','budget':'达到四次上限'}
    for i,(o,n) in enumerate(selected,1):
        decision=n['events'][n['steps']-1]['decision']
        second=('采信' if n['events'][1]['accepted'] else '拒收') if n['steps']>=2 else '未调用'
        print(f"| {i:02} | {abbr[o['first']]} → {abbr[o['next']]} | {'采信' if o['bounded']['accepted'] else '拒收'} | "
              + ' → '.join(map(str,o['grades']))+f" | {n['old']} | {n['steps']} | {second} | {n['revised']} | {reason[decision]} |")
else:
    summary={'old_prompt_stop_counts':{k:sum(n['events'][0]['replay'][k] for o,n in selected)
                                      for k in selected[0][1]['events'][0]['replay']},
             'first_reasons':dict(Counter(n['events'][0]['decision'] for o,n in selected)),
             'next_tool':{},'accepted_first':{}}
    for key,values in [('next_tool',set(o['next'] for o,n in selected)),('accepted_first',{True,False})]:
        for v in values:
            group=[(o,n) for o,n in selected if (o['next'] if key=='next_tool' else o['bounded']['accepted'])==v]
            summary[key][str(v)]={'n':len(group),'continued':sum(n['steps']>1 for o,n in group),
                                  'all_evidence_first_to_second':moves([o['grades'][0] for o,n in group],[o['grades'][1] for o,n in group]),
                                  'revised_correct':sum(n['revised'] for o,n in group)}
    rescued=[n for o,n in selected if o['baseline']>o['grades'][0]]
    summary['baseline_final_rescues']=dict(n=len(rescued),continued=sum(n['steps']>1 for n in rescued),
            accepted_second=sum(n['steps']>=2 and n['events'][1]['accepted'] for n in rescued),
            revised_correct=sum(n['revised'] for n in rescued),control_correct=sum(n['control'] for n in rescued))
    print(json.dumps(summary,indent=2))
