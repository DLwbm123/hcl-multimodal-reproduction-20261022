"""Resumable official downloads; hard quota is enforced by the parent supervisor."""
import concurrent.futures
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path
ROOT=Path(os.environ['REPRO_ROOT'])
RAW=ROOT/'raw'
FILES={
 'train2014.zip':'https://s3.amazonaws.com/images.cocodataset.org/zips/train2014.zip',
 'val2014.zip':'https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip',
 'annotations_trainval2014.zip':'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip',
 'vqa_questions.zip':'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
 'vqa_annotations.zip':'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
 'vqa_train_questions.zip':'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
 'vqa_train_annotations.zip':'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
}

def fetch(item):
 name,url=item;p=RAW/name
 with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r: expected=int(r.headers['Content-Length'])
 if p.exists() and p.stat().st_size==expected:
  return {'name':name,'bytes':expected,'source':url}
 partial=p.with_suffix('.partial');receipt=p.with_suffix('.ranges.json');block=8*2**20
 completed=set()
 if receipt.exists():completed=set(json.loads(receipt.read_text())['completed'])
 elif p.exists():
  prefix=p.stat().st_size
  p.rename(partial)
  completed=set(range(prefix//block))
  partial.open('r+b').truncate((prefix//block)*block)
 fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o644)
 def part(index):
  start=index*block;end=min(start+block,expected)-1
  for attempt in range(5):
   try:
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=90) as r:
     if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise ValueError('Unexpected range response')
     data=r.read(end-start+2)
    if len(data)!=end-start+1:raise ValueError('Incomplete range')
    written=os.pwrite(fd,data,start)
    if written!=len(data):raise OSError('Incomplete pwrite')
    return index
   except Exception:
    if attempt==4:raise
    time.sleep(3)
 try:
  missing=[i for i in range((expected+block-1)//block) if i not in completed]
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
   for i in ex.map(part,missing):
    completed.add(i)
    tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps({'completed':sorted(completed),'expected':expected,'block_size':block}));tmp.replace(receipt)
  os.fsync(fd)
 finally:os.close(fd)
 if partial.stat().st_size!=expected:raise ValueError('Final length mismatch')
 partial.replace(p)
 print(json.dumps({'downloaded':name,'bytes':expected,'source':url}),flush=True)
 return {'name':name,'bytes':expected,'source':url}

def main():
 RAW.mkdir(parents=True,exist_ok=True)
 # Existing partial files must no longer be written by another downloader.
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e: receipts=list(e.map(fetch,FILES.items()))
 image_root=ROOT/'images';image_root.mkdir(exist_ok=True)
 for name in ['train2014.zip','val2014.zip']:
  with zipfile.ZipFile(RAW/name) as z:
   expected=sum(i.file_size for i in z.infolist())
   print(json.dumps({'extract':name,'uncompressed_bytes':expected}),flush=True)
   for i in z.infolist():
    if i.is_dir():continue
    destination=image_root/Path(i.filename).name
    if destination.exists() and destination.stat().st_size==i.file_size:continue
    temporary=destination.with_suffix('.extract')
    with z.open(i) as src,temporary.open('wb') as dst:
     import shutil
     shutil.copyfileobj(src,dst,2**20)
    temporary.replace(destination)
  print(json.dumps({'extracted':name}),flush=True)
 ROOT.joinpath('download_receipts.json').write_text(json.dumps(receipts,indent=2))
 ROOT.joinpath('DOWNLOAD_COMPLETE').write_text('official COCO train/val and VQAv2 train/val; RefCOCO archive supplied separately\n')

if __name__=='__main__':main()
