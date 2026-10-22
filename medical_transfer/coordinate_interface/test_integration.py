"""Actual pinned pipeline and multimodal caller, fake weights only."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace,MethodType
from PIL import Image
import torch
from integration import CoordinateBoundary,install_coordinate_boundary,construct_coordinate_pipeline
from hcl.models import LocalTransformersModel,RuleBasedSmokeModel
from hcl.task_interface import TaskInterfaceConfig
from medical_hcl.bridge import construct_pipeline

class Inputs(dict):
    input_ids=torch.zeros((1,2),dtype=torch.long)
    def to(self,device):return self
class Processor:
    def __init__(self):
        self.image_processor=SimpleNamespace(patch_size=14,merge_size=2,min_pixels=3136,max_pixels=12845056,resample=Image.Resampling.BICUBIC)
        self.rendered=[];self.calls=[]
    def apply_chat_template(self,messages,**kwargs):
        self.rendered.append(messages);return str(messages)
    def __call__(self,*,text,images,do_resize,return_tensors):
        self.calls.append((images[0].size,do_resize))
        w,h=images[0].size
        return Inputs(image_grid_thw=torch.tensor([[1,h//14,w//14]]))
    def batch_decode(self,*args,**kwargs):return ['```json\n{"bbox_2d":[28,49,84,147]}\n```']
class Weights:
    device='cpu'
    def generate(self,**kwargs):return torch.zeros((1,4),dtype=torch.long)
class Backend(LocalTransformersModel):
    def __init__(self):
        super().__init__(name='fixture',path='unused',max_new_tokens=512,answer_max_new_tokens=512)
        self._processor=Processor();self._model=Weights();self.delegated=[]
    def _load(self):pass
    def generate(self,prompt,*,state):self.delegated.append((prompt,state));return 'UNCHANGED'

class WiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.image=self.root/'anonymous.png';Image.new('RGB',(100,200)).save(self.image)
        self.backend=Backend();self.boundary=CoordinateBoundary(self.backend)
        self.example={'task_id':'case','id':'case','task_name':'polyp_localization','task_type':'medical_polyp_localization','image':str(self.image),'question':'old original-image instruction','visible_context':{'width':100,'height':200},'answer':{'bbox':[0,0,100,200]},'metadata':{'width':100,'height':200,'role':'validation','group_id':'g','pixel_id':'p','secret':'GOLD_CANARY_PRIVATE'},'split':'validation'}
    def state(self):return {'internal_sample_id':'case','question':'完全不同的问句','task_context':{'files':[{'type':'image','path':str(self.image)}]}}
    def test_actual_caller_grid_mapping_and_no_gold(self):
        adapted=self.boundary.prepare_example(self.example)
        self.assertEqual(self.example['question'],'old original-image instruction')
        self.assertNotIn('original-image instruction',adapted['question'])
        answer=json.loads(self.boundary.generate('rewritten H with no old decoder trigger',state=self.state()))
        self.assertEqual(answer,{'bbox':[25,50,75,150]})
        self.assertEqual(self.backend._processor.calls,[((112,196),False)])
        self.assertNotIn('GOLD_CANARY_PRIVATE',str(self.backend._processor.rendered))
        self.assertNotIn('GOLD_CANARY_PRIVATE',str(self.boundary.frames))
    def test_roles_and_nonlocalization_passthrough(self):
        self.boundary.prepare_example(self.example)
        state={**self.state(),'router_phase':'select_memory'}
        self.assertEqual(self.boundary.generate('memory JSON',state=state),'UNCHANGED')
        for task in ['classification','visual_qa']:
            e={**self.example,'task_name':task,'task_id':task}
            self.assertEqual(self.boundary.prepare_example(e),e)
            self.assertEqual(self.boundary.generate('other',state={'internal_sample_id':task}),'UNCHANGED')
    def test_binding_cannot_change_or_use_different_image(self):
        self.boundary.prepare_example(self.example)
        other=self.root/'other.png';Image.new('RGB',(100,200)).save(other)
        with self.assertRaises(ValueError):self.boundary.prepare_example({**self.example,'image':str(other)})
        s=self.state();s['task_context']['files'][0]['path']=str(other)
        with self.assertRaises(ValueError):self.boundary.generate('x',state=s)
    def test_invalid_bbox_is_failed_answer_not_exception_or_drop(self):
        self.boundary.prepare_example(self.example)
        self.backend._processor.batch_decode=lambda *a,**kw:['{"bbox":[0,0,1000,2000]}']
        self.assertEqual(self.boundary.generate('x',state=self.state()),'')
        self.assertFalse(self.boundary.last_event['format_valid'])
    def test_shared_factory_freshness_and_deterministic_gate(self):
        prepared=self.root/'factory_data';prepared.mkdir()
        (prepared/'FROZEN.json').write_text('{"seed":20261022}')
        (prepared/'class_vocabulary.json').write_text('["normal","polyp"]')
        for arm in ['static_harness','stability_hcl']:
            storage=self.root/('factory_'+arm)
            pipe,boundary=construct_coordinate_pipeline(prepared,storage,backend=Backend(),arm=arm)
            self.assertTrue(pipe._coordinate_boundary_installed)
            self.assertIs(pipe.model,boundary)
            self.assertEqual(pipe.optimizer.min_primary_score_delta,1e-9)
            self.assertEqual(pipe.optimizer.min_format_compliance_rate,1.0)
            (storage/'checkpoint.json').write_text('{}')
            with self.assertRaises(ValueError):construct_coordinate_pipeline(prepared,storage,backend=Backend(),arm=arm)
        backend=Backend();backend.config.temperature=.1
        with self.assertRaises(ValueError):construct_coordinate_pipeline(prepared,self.root/'bad_temperature',backend=backend,arm='static_harness')

    def test_hcl_memory_frame_tag_and_rollback_keep_original_feedback(self):
        prepared=self.root/'memory_data';prepared.mkdir()
        (prepared/'FROZEN.json').write_text('{"seed":20261022}')
        (prepared/'class_vocabulary.json').write_text('["normal","polyp"]')
        pipe,boundary=construct_coordinate_pipeline(prepared,self.root/'memory_run',backend=Backend(),arm='stability_hcl')
        e=copy.deepcopy(self.example);e['metadata']['role']='h_search';e['split']='h_search'
        adapted=boundary.prepare_example(e)
        before=copy.deepcopy(pipe.memory.snapshot())
        pipe.memory.record_training_interaction('polyp_localization',adapted,{'task_id':'case','answer':'{"bbox":[25,50,75,150]}'})
        record=pipe.memory.raw.records[0]
        self.assertIn('original image 100 x 200',record['question'])
        self.assertIn('not coordinates for the currently supplied image',record['question'])
        self.assertEqual(record['feedback'],{'bbox':[0,0,100,200]})
        self.assertFalse(record['response_is_correct'])
        pipe.memory.restore(before);self.assertEqual(pipe.memory.snapshot(),before)
        adapted['metadata']['role']='retention_anchor'
        with self.assertRaises(ValueError):pipe.memory.record_training_interaction('polyp_localization',adapted,{'task_id':'case','answer':''})

    def test_actual_hcl_pipeline_both_modes_and_readonly(self):
        prepared=self.root/'prepared';prepared.mkdir();(prepared/'FROZEN.json').write_text('{"seed":20261022}')
        (prepared/'class_vocabulary.json').write_text('["normal","polyp"]')
        payloads=[]
        for arm in ['static_harness','stability_hcl']:
            backend=Backend();boundary=CoordinateBoundary(backend)
            pipe=construct_pipeline(prepared,self.root/arm,model=boundary,real_execution_authorized=True,arm=arm)
            pipe.task_interface_config=TaskInterfaceConfig(mode='rule')
            pipe.selection_model=RuleBasedSmokeModel()
            pipe.router_config.use_chat_tool_calls=False;pipe.router_config.use_llm_workflow_selector=False
            pipe.router_config.use_memory_selector=False;pipe.router_config.memory_always_on=False
            pipe.router_config.use_skill_selector=False
            install_coordinate_boundary(pipe,boundary)
            before=(copy.deepcopy(pipe.memory.snapshot()),copy.deepcopy(pipe.component_versions),pipe.harness_version,len(pipe.optimizer.history))
            pred=pipe._predict([self.example],update_memory=False,progress_label='cpu-dev-only')
            self.assertEqual(json.loads(pred[0]['answer']),{'bbox':[25,50,75,150]})
            after=(pipe.memory.snapshot(),pipe.component_versions,pipe.harness_version,len(pipe.optimizer.history))
            self.assertEqual(before,after)
            messages=backend._processor.rendered[-1]
            pixels=messages[1]['content'][0]['image']
            payloads.append((messages[0]['content'],messages[1]['content'][1]['text'],pixels.size,pixels.tobytes()))
            with self.assertRaises(ValueError):install_coordinate_boundary(pipe,boundary)
        self.assertEqual(payloads[0],payloads[1])

if __name__=='__main__':unittest.main()
