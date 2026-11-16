"""Gold labels are accessible only here and in post-episode training feedback."""
import math
import numpy as np
from PIL import Image
from core import bbox

def iou(a,b):
 intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
 union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
 return intersection/union if union else 0.

def score(task,case,episode,tools,p):
 ans=episode['answer'];r={'score':0.,'success':False,'valid':False,'bbox_iou':0.,'dice':0.,'area_error':1.,'area_consistency_error':1.}
 try:
  if episode.get('error'):raise ValueError(episode['error'])
  if task=='T1':
   if set(ans)!={'bbox'}:raise ValueError('T1 exact answer schema')
   b=bbox(ans['bbox'],case['width'],case['height']);v=iou(b,case['bbox']);r.update(score=v,bbox_iou=v,success=v>=p['T1_success_iou'],valid=True)
  else:
   if set(ans)!={'mask_id','area_fraction'}:raise ValueError('T2 exact answer schema')
   mask=tools.masks[ans['mask_id']];fraction=ans['area_fraction']
   if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('area fraction 0..1 required')
   gt=np.asarray(Image.open(case['mask']).convert('L'))>127
   if mask.shape!=gt.shape:raise ValueError('mask size mismatch')
   dice=2*int((mask & gt).sum())/max(1,int(mask.sum()+gt.sum()));err=abs(float(gt.mean())-fraction)
   consistency=abs(float(mask.mean())-fraction)
   r.update(score=dice,dice=dice,area_error=err,area_consistency_error=consistency,success=dice>=p['T2_success_dice'] and err<=p['T2_success_area_abs_error'] and consistency<=p.get('T2_area_consistency_tolerance',.001),valid=True)
 except (ValueError,KeyError,TypeError) as e:r['invalid_reason']=str(e)
 return r

def aggregate(rows):
 n=len(rows)
 if not n:return {'n':0,'mean_score':0.,'correct':0,'valid':0}
 return {'n':n,'mean_score':float(np.mean([r['metrics']['score'] for r in rows])),'correct':sum(r['metrics']['success'] for r in rows),'valid':sum(r['metrics']['valid'] for r in rows),'skill_episodes':sum(bool(r['episode']['skills_used']) for r in rows),'mean_primitive_calls':float(np.mean([len(r['episode']['tools']) for r in rows])),'mean_vlm_calls':float(np.mean([r['episode']['vlm_calls'] for r in rows])),'mean_sam_calls':float(np.mean([r['episode']['sam_calls'] for r in rows])),'mean_seconds':float(np.mean([r['episode']['seconds'] for r in rows]))}

def accept(current,candidate,old_before,old_after,g):
 a=aggregate(current);b=aggregate(candidate);reasons=[]
 quality_gain=b['mean_score']-a['mean_score']>=g['min_current_mean_gain']-1e-12 and b['correct']-a['correct']>=g['min_current_success_gain']
 efficiency_gain=bool(g.get('allow_efficiency_gain')) and b['mean_score']>=a['mean_score']-1e-12 and b['correct']>=a['correct'] and b['mean_primitive_calls']<=a['mean_primitive_calls']-g['min_primitive_reduction'] and b['mean_vlm_calls']<=a['mean_vlm_calls']*(1-g['min_vlm_fraction_reduction'])
 if not (quality_gain or efficiency_gain):reasons.append('neither_quality_nor_efficiency_gain')
 if b['valid']<a['valid']:reasons.append('invalid_outputs_increased')
 if b['skill_episodes']==0:reasons.append('no_skill_invocation')
 losses=0
 if old_before:
  assert [r['id'] for r in old_before]==[r['id'] for r in old_after]
  losses=sum(a['metrics']['success'] and not b['metrics']['success'] for a,b in zip(old_before,old_after))
  if losses>g['max_historical_success_losses']:reasons.append('historical_success_losses')
  if aggregate(old_after)['mean_score']<aggregate(old_before)['mean_score']-g['max_historical_mean_drop']-1e-12:reasons.append('historical_mean_drop')
  if aggregate(old_after)['valid']<aggregate(old_before)['valid']:reasons.append('historical_invalid_outputs_increased')
 return {'accepted':not reasons,'reasons':reasons,'gain_type':'quality' if quality_gain else 'efficiency' if efficiency_gain else 'none','current_before':a,'current_after':b,'historical_losses':losses,'history_before':aggregate(old_before),'history_after':aggregate(old_after)}
