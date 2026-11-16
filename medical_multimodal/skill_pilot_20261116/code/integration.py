"""Explicit opt-in coordinate boundary for the pinned HCL pipeline, not a global patch."""
import copy
import json
from pathlib import Path
from types import MethodType
from geometry import FrameContract

class CoordinateBoundary:
    def __init__(self, backend, audit_path=None):
        self.backend = backend
        self.config = backend.config
        self.frames = {}
        self.audit_path = Path(audit_path) if audit_path else None
        self.last_event = None

    def prepare_example(self, example):
        if example.get('task_name') != 'polyp_localization': return copy.deepcopy(example)
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        from PIL import Image
        self.backend._load()
        proc = self.backend._processor.image_processor
        # Only visible image metadata enters the immutable registry; never gold or role.
        with Image.open(example['image']) as image: w, h = image.size
        context = example['visible_context']
        if (context['width'], context['height']) != (w, h):
            raise ValueError('image/visible metadata mismatch')
        ih, iw = smart_resize(h, w, factor=proc.patch_size*proc.merge_size,
                             min_pixels=proc.min_pixels,max_pixels=proc.max_pixels)
        contract = FrameContract(w,h,iw,ih)
        key = str(example['task_id']); bound = (str(Path(example['image']).resolve()), contract)
        if key in self.frames and self.frames[key] != bound: raise ValueError('immutable binding changed')
        self.frames[key] = bound
        result = copy.deepcopy(example)
        # Old original-file coordinate instructions are replaced before any H rewriting.
        result['question'] = example.get('coordinate_goal', 'Return one tight bounding box enclosing the visible polyp region(s).')+' '+contract.instruction()
        result['visible_context'] = contract.visible_contract()
        return result

    def generate(self, prompt, *, state):
        from hcl.models import _state_phase
        key = str(state.get('internal_sample_id', state.get('task_id','')))
        if _state_phase(state) != 'final_answer' or key not in self.frames:
            return self.backend.generate(prompt,state=state)
        image_path, frame = self.frames[key]
        actual = self.backend._image_path_from_state(state)
        if not actual or str(Path(actual).resolve()) != image_path:
            raise ValueError('final image does not match immutable binding')
        self.backend._load()
        from PIL import Image
        processor = self.backend._processor; model = self.backend._model
        with Image.open(image_path) as image:
            supplied = image.convert('RGB').resize((frame.input_width,frame.input_height),
                                                   resample=processor.image_processor.resample)
        # Runtime contract is outside learnable H. It is not recovered from rewritten text.
        messages=[{'role':'system','content':self.config.system_prompt+'\nImmutable runtime image/output contract (takes precedence over harness text): '+frame.instruction()},
                  {'role':'user','content':[{'type':'image','image':supplied},{'type':'text','text':prompt+'\nRuntime contract: '+frame.instruction()}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                                            enable_thinking=self.config.enable_thinking)
        inputs=processor(text=[text],images=[supplied],do_resize=False,return_tensors='pt')
        grid=inputs['image_grid_thw'].tolist()
        expected=[[1,frame.input_height//processor.image_processor.patch_size,
                     frame.input_width//processor.image_processor.patch_size]]
        if grid!=expected: raise ValueError('actual processor grid mismatches immutable frame')
        inputs=inputs.to(model.device)
        limit=min(self.config.max_new_tokens,self.config.answer_max_new_tokens)
        if state.get('max_new_tokens') is not None:limit=min(limit,max(1,int(state['max_new_tokens'])))
        import torch
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=limit,do_sample=False)
        raw=processor.batch_decode(output[:,inputs.input_ids.shape[-1]:],skip_special_tokens=True)[0].strip()
        normalized=''; error=None
        try: normalized=frame.normalize_answer(raw)
        except (ValueError,TypeError,KeyError) as exc: error=str(exc)
        # An invalid output is a failed answer in the original evaluator, never dropped.
        event={'sample_id':key,'raw':raw,'normalized':normalized,'format_valid':error is None,
               'failure_reason':error,'input_width':frame.input_width,'input_height':frame.input_height,
               'original_width':frame.original_width,'original_height':frame.original_height,
               'actual_grid':grid,'final_prompt':prompt,'runtime_contract':frame.instruction()}
        self.last_event=event
        if self.audit_path:
            with self.audit_path.open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
        return normalized


def install_coordinate_boundary(pipeline,boundary):
    """Call once on a NEW pipeline with isolated caches; never on a resumed old run."""
    if pipeline.model is not boundary: raise ValueError('shared boundary must be the pipeline model')
    if getattr(pipeline,'_coordinate_boundary_installed',False): raise ValueError('already installed')
    if pipeline.checkpoint.has_state or pipeline.global_batch_index or pipeline.harness_version or any(int(v.get('version',0)) for v in pipeline.component_versions.values()):
        raise ValueError('cannot migrate an existing learned run in place')
    if pipeline.read_only_workers != 1:raise ValueError('coordinate backend requires serialized model calls')
    if pipeline.router_config.use_chat_tool_calls:raise ValueError('tool generation path not supported by this protocol')
    if pipeline.router_config.final_generation_cache_path or pipeline.prediction_cache.enabled:
        raise ValueError('coordinate protocol requires fresh disabled prediction/final caches')
    original=pipeline._predict_with_router
    def prepared(self,examples,router,**kwargs):
        return original([boundary.prepare_example(e) for e in examples],router,**kwargs)
    pipeline._predict_with_router=MethodType(prepared,pipeline)
    original_add=pipeline.memory.raw.add
    def tagged_memory(self,task_name,examples,predictions,**kwargs):
        tagged=[]
        for example in examples:
            item=copy.deepcopy(example)
            if item.get('task_name')=='polyp_localization':
                path,frame=boundary.frames[str(item['task_id'])]
                # The evaluator and stored feedback stay in original pixels. Label that frame
                # explicitly so retrieved normalized answers are not presented as input pixels.
                item['question']=(f'Historical polyp-localization case, original image '
                    f'{frame.original_width} x {frame.original_height}. Stored response and feedback '
                    'boxes use this historical original-image half-open xyxy pixel frame. '
                    'These are not coordinates for the currently supplied image.')
            tagged.append(item)
        return original_add(task_name,tagged,predictions,**kwargs)
    pipeline.memory.raw.add=MethodType(tagged_memory,pipeline.memory.raw)
    pipeline._coordinate_boundary_installed=True
    return pipeline


def construct_coordinate_pipeline(prepared,storage,*,backend,arm):
    """Single opt-in factory for BOTH future arms; no execution or data split access."""
    from medical_hcl.bridge import construct_pipeline
    storage=Path(storage)
    if (storage/'checkpoint.json').exists():raise ValueError('fresh storage required; do not reinterpret old checkpoints')
    if backend.config.temperature != 0:raise ValueError('this frozen contract uses deterministic decoding')
    boundary=CoordinateBoundary(backend,storage/'private_interface_trace.jsonl')
    pipeline=construct_pipeline(prepared,storage,model=boundary,real_execution_authorized=True,arm=arm)
    # Current protocol also disables tool generation. All other selectors/gates stay as configured.
    pipeline.router_config.use_chat_tool_calls=False
    install_coordinate_boundary(pipeline,boundary)
    return pipeline,boundary
