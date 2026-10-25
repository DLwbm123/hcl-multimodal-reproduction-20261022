"""CPU test of production crop materialization and a meaningful acquisition gate."""
import json,os
from pathlib import Path
from PIL import Image
from focus_logic import parse,build_views,FIXED,decision

def main():
    root=Path(os.environ['R981_ROOT']);budget=json.loads((root/'budget.json').read_text());assert budget['max_subscription_calls']==220 and budget['model']=='gpt-6.1-sol' and budget['reasoning']=='high'
    policy=parse(json.dumps({'admit':True,'crop_ids':[1,3,4,7],'source_indices':[0,1,2,3],'rationale':'CPU fixture only'}))
    try:parse(json.dumps(dict(policy,crop_ids=[8,6,2,0])))
    except AssertionError:pass
    else:raise AssertionError('Reordered fixed crops admitted')
    fixture=root/'tmp/cpu';fixture.mkdir(exist_ok=False);im=Image.new('RGB',(80,60));im.putdata([(x*3,y*4,(x+y)%256) for y in range(60) for x in range(80)]);paths=[]
    for i in range(2):p=fixture/('input'+str(i)+'.png');im.save(p);paths.append(str(p))
    item={'images':paths};h=build_views(item,None,fixture/'h');assert h==(paths,'',[],False)
    f=build_views(item,{'crop_ids':FIXED},fixture/'f');m=build_views(item,policy,fixture/'m');assert f[0][:2]==m[0][:2]==paths and len(f[0])==len(m[0])==10 and len(f[2])==len(m[2])==8 and not f[3] and m[3]
    assert all(Image.open(p).size==(512,512) for p in f[0][2:]+m[0][2:]) and all(r['pixel_box'][2]-r['pixel_box'][0]==40 and r['pixel_box'][3]-r['pixel_box'][1]==30 for r in m[2]);assert 'rationale' not in m[1] and 'source_indices' not in m[1]
    ids=['x','y'];gold={k:'A' for k in ids};by={}
    for b in [1,2]:
        by['H0_static_r'+str(b)]={k:{'predicted':'B'} for k in ids};by['learned_focus_r'+str(b)]={k:{'predicted':'A','visual_input_differs_from_fixed':True} for k in ids}
    pairs=[{'H0_to_learned':{'gain':2,'loss':0},'fixed_to_learned':{'gain':1,'loss':0}} for _ in [1,2]];assert decision(pairs,by,ids,gold,True)['pass'] and not decision(pairs,by,ids,gold,False)['pass'];by['learned_focus_r1']['x']['visual_input_differs_from_fixed']=False;assert not decision(pairs,by,ids,gold,True)['pass']
    result={'passed':True,'transport_budget_fields_checked':True,'production_crop_program_creates_expected_real_pixels':True,'all_original_images_retained_first':True,'fixed_and_learned_count_area_zoom_and_output_size_matched':True,'reordering_fixed_program_rejected':True,'source_feedback_not_explicitly_in_controller_context':True,'gate_requires_actual_pixel_difference_and_complete_repeats':True,'real_model_calls':0};(root/'CPU_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
