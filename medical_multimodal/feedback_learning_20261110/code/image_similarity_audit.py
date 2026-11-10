"""Image-only grouping audit triggered by observed cross-ID duplicate images.

No benchmark question, answer, explanation or clinical metadata is opened.
High similarity is a candidate for review, not proof of patient identity.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image

root=Path(os.environ['AUDIT_ROOT']); images=Path(os.environ['IMAGE_ROOT'])
manifest=json.loads((root/'image_manifest.private.json').read_text())
selected={x['relative_path']:x for x in manifest}
paths=sorted(p for p in images.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})
features=[]; names=[]; errors=[]
started=time.time()
for p in paths:
 try:
  with Image.open(p) as im:
   a=np.asarray(im.convert('L').resize((64,64)),dtype=np.float32).reshape(-1)
  centered=a-a.mean(); norm=np.linalg.norm(centered)
  if norm<1:continue
  features.append(centered/norm);names.append(str(p.relative_to(images)))
 except Exception as e:errors.append({'path':str(p.relative_to(images)),'error':type(e).__name__})
a=np.stack(features); lookup={n:i for i,n in enumerate(names)};out=[]
for rel,item in selected.items():
 if rel not in lookup:raise FileNotFoundError(rel)
 i=lookup[rel];similarity=a@a[i]; source=Path(rel).parts[1]
 for j in np.flatnonzero(similarity>=0.985):
  other=names[j]
  if Path(other).parts[1]==source:continue
  out.append({'index':item['index'],'selected_path':rel,'other_path':other,'correlation64':float(similarity[j]),'other_reserved':other in selected,'other_index':selected.get(other,{}).get('index')})
report={'trigger':'Two visually identical cross-source image pairs observed; both pairs had 128px thumbnail MAD zero.','scope':'198 reserved images compared with all existing benchmark image files; no labels or question text read','selected_images':len(selected),'compared_files':len(paths),'usable_files':len(names),'errors':errors,'threshold':0.985,'elapsed_seconds':time.time()-started,'candidates':out,'limitation':'Aligned grayscale thumbnail screen can miss cropped, rotated, mirrored or recompressed duplicates; similarity is not patient identity.'}
(root/'similarity_candidates.private.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='candidates'}));print('candidate_pairs',len(out))
