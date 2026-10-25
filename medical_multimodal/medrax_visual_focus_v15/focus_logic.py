"""A finite source-learned geometric program changes actual current-image views."""
import json,re
from pathlib import Path
ARMS=['H0_static','fixed_focus','learned_focus']
FINAL='If you had to choose the best option, only respond with the letter of choice (only one of A, B, C, D, E, F)'
# Nine equal-area windows: all half-width by half-height, fixed2x zoom.
CATALOG={i:[x,y,x+0.5,y+0.5] for i,(x,y) in enumerate((x,y) for y in [0.,0.25,0.5] for x in [0.,0.25,0.5])}
FIXED=[0,2,6,8]

def parse(text):
    s=text.strip()
    if s.startswith('```'):s=re.sub(r'^```(?:json)?\s*|\s*```$','',s)
    v=json.loads(s);assert set(v)=={'admit','crop_ids','source_indices','rationale'} and type(v['admit']) is bool and isinstance(v['rationale'],str) and 0<len(v['rationale'])<=1600
    if not v['admit']:return v
    assert isinstance(v['crop_ids'],list) and len(v['crop_ids'])==4 and all(type(i) is int and i in CATALOG for i in v['crop_ids']) and len(set(v['crop_ids']))==4
    assert len(set(v['crop_ids'])-set(FIXED))>=2,'Need a real region-selection change, not a reordering of fixed quadrants'
    assert isinstance(v['source_indices'],list) and 4<=len(v['source_indices'])<=24 and all(type(i) is int and 0<=i<24 for i in v['source_indices']) and len(set(v['source_indices']))==len(v['source_indices'])
    v['crop_ids']=sorted(v['crop_ids']);return v

def crop(image,index):
    box=CATALOG[index];w,h=image.size;pixel=(round(box[0]*w),round(box[1]*h),round(box[2]*w),round(box[3]*h));assert pixel[2]>pixel[0] and pixel[3]>pixel[1]
    return image.crop(pixel).resize((512,512)),pixel

def build_views(item,policy,directory):
    if policy is None:return list(item['images']),'',[],False
    from PIL import Image,ImageChops
    directory=Path(directory);directory.mkdir(exist_ok=False);paths=list(item['images']);rows=[];different=False
    for j,path in enumerate(item['images']):
        with Image.open(path) as raw:
            image=raw.convert('RGB')
            for k,index in enumerate(policy['crop_ids']):
                part,pixel=crop(image,index);reference,_=crop(image,FIXED[k]);different=different or ImageChops.difference(part,reference).getbbox() is not None
                dest=directory/('p'+str(j+1)+'_'+str(k+1)+'.png');part.save(dest);paths.append(str(dest));rows.append({'current_image_index':j+1,'normalized_box':CATALOG[index],'pixel_box':pixel,'original_size':list(image.size),'view_path':str(dest),'output_size':[512,512]})
    note='\nAdditional zoom views were DERIVED from the same CURRENT images. All original images appear first, followed by four equal-area crops per original image in the mapping below. These are not independent images, patients, timepoints or new clinical observations. Use the originals to retain global context and orientation; cropping cannot establish absent chronology, cause, measurements or diagnoses.\n'+json.dumps(rows,ensure_ascii=False)
    return paths,note,rows,bool(different)

def decision(pairs,by,ids,gold,complete):
    acquired=[]
    for k in ids:
        if all(by['H0_static_r'+str(b)][k].get('predicted')!=gold[k] and by['learned_focus_r'+str(b)][k].get('predicted')==gold[k] and by['learned_focus_r'+str(b)][k].get('visual_input_differs_from_fixed',False) for b in [1,2]):acquired.append(k)
    passed=complete and len(acquired)>=2 and all(p['H0_to_learned']['gain']-p['H0_to_learned']['loss']>=2 and p['H0_to_learned']['loss']<=1 and p['fixed_to_learned']['gain']-p['fixed_to_learned']['loss']>=1 for p in pairs)
    return {'pass':bool(passed),'repeated_gain_with_actual_different_pixels_count':len(acquired),'repeated_gain_ids':acquired,'exploratory_error_enriched_benchmark_only':True}
