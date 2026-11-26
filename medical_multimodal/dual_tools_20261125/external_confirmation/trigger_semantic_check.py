"""Aggregate supplementary blind grades for every triggered primary pair."""
import sys,json,random,pathlib
p=pathlib.Path(sys.argv[1])
grades=json.loads((p/'private_third_grades.json').read_text())
result={'n':0,'original_correct':0,'review_correct':0,'gains':0,'losses':0,'unresolved':0}
seen=set(); detail=[]
for f in sorted(p.glob('part?/private_generated.json')):
 for k,r in json.load(open(f)).items():
  if not r['trigger']:continue
  seen.add(k); a=r['answers']; vals=sorted(set((a['V'],a['S']))); random.Random(661+int(k)).shuffle(vals)
  g=grades[k]; assert len(g)==len(vals)
  marks=dict(zip(vals,g)); s=marks[a['S']];v=marks[a['V']]
  result['n']+=1;result['original_correct']+=s=='1';result['review_correct']+=v=='1'
  result['gains']+=s=='0' and v=='1';result['losses']+=s=='1' and v=='0'
  result['unresolved']+='U' in (s,v)
  detail.append([k,s,v])
assert seen==set(grades)
(p/'private_third_cases.json').write_text(json.dumps(detail))
print(json.dumps(result,indent=2))
