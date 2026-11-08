import os,time,json,subprocess,tempfile,shutil,concurrent.futures
from pathlib import Path
root=Path(os.environ['R25_ROOT']);prior=Path(os.environ['R25_PRIOR']);jobs=json.load(open(prior/'private/jobs.json'));empty=[i for i in range(6) if json.load(open(prior/'private'/('ocr_%02d.json'%i)))['characters']==0];assert empty==[1,2,3,4,5]
def go(i):
 with tempfile.TemporaryDirectory(prefix='r25-',dir='/tmp') as td:
  p=Path(td)/'input.jpg';shutil.copyfile(prior/'private'/jobs[i]['image_name'],p)
  try:
   out=subprocess.run(['/opt/homebrew/bin/tesseract',str(p.resolve()),'stdout','--psm','11'],capture_output=True,encoding='utf-8',errors='replace',timeout=25)
   (root/'private'/('ocr_%02d.json'%i)).write_text(json.dumps({'text':out.stdout,'exit':out.returncode}))
   return {'case_index':i,'exit':out.returncode,'characters':len(out.stdout.strip())}
  except subprocess.TimeoutExpired:return {'case_index':i,'exit':'timeout','characters':None}
start=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:r=list(ex.map(go,empty))
summary={'time':time.time(),'source':'previously exposed v19 cases with zero default OCR','planned_calls':5,'completed_calls':len(r),'exit_zero':sum(x['exit']==0 for x in r),'new_nonempty':sum(isinstance(x['characters'],int) and x['characters']>0 for x in r),'still_empty':sum(x['characters']==0 for x in r),'timed_out':sum(x['exit']=='timeout' for x in r),'elapsed_seconds':time.time()-start,'interpretation':'Output presence is not OCR accuracy; post-hoc diagnostic, no answer labels or scores; not a model evaluation.'}
(root/'AGGREGATE.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
