"""Asset/path and measurement adapters; native tools and agent own execution."""
import importlib.util
import json
from pathlib import Path
import sys
import time


def load_module(path, name, replacements=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if replacements:
        source = path.read_text()
        for old, new in replacements.items():
            if old not in source:
                raise RuntimeError('Pinned upstream asset reference changed')
            source = source.replace(old, new)
        exec(compile(source, str(path), 'exec'), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


class BoundedTool:
    def __init__(self, tool, state, record, guard):
        self.tool, self.state, self.record, self.guard = tool, state, record, guard
        self.name, self.description, self.args_schema = tool.name, tool.description, tool.args_schema

    def invoke(self, args):
        self.guard()
        self.state['calls'] += 1
        if self.state['calls'] > self.state['max_calls']:
            raise RuntimeError('Episode tool-call budget exhausted')
        start = time.time()
        try:
            parsed = self.args_schema.model_validate(args).model_dump()
            paths = parsed.get('image_paths', [parsed.get('image_path')])
            if not paths or any(str(Path(p).resolve()) not in self.state['images'] for p in paths):
                raise ValueError('Tool image path is outside this episode')
            if not 1 <= parsed.get('max_new_tokens', 512) <= 512:
                raise ValueError('Expert output must be between 1 and 512 tokens')
        except Exception as exc:
            result = ({'error': str(exc)}, {'analysis_status': 'failed'})
            self.record('tool_finished', {'name': self.name, 'args': args,
                'status': 'rejected_input', 'seconds': time.time() - start, 'result': str(result)})
            return result
        self.record('tool_started', {'name': self.name, 'args': args})
        result = self.tool.invoke(args)
        self.record('tool_finished', {'name': self.name, 'args': args,
            'status': result[1].get('analysis_status', 'unknown') if isinstance(result, tuple) else 'unknown',
            'seconds': time.time() - start, 'result': str(result)})
        return result


def load_native(root, record, guard):
    import torchxrayvision as xrv
    import transformers
    import torch
    tools_dir = root / 'upstream/medrax/tools'
    # xrv resolves both public checkpoints through this shared cache helper.
    xrv.utils.get_cache_dir = lambda: str(root / 'cache/torchxrayvision')
    classifier = load_module(tools_dir / 'classification.py', 'native_classification').ChestXRayClassifierTool(device='cuda:0')
    segment = load_module(tools_dir / 'segmentation.py', 'native_segmentation').ChestXRaySegmentationTool(
        device='cuda:0', temp_dir=root / 'outputs')
    report = load_module(tools_dir / 'report_generation.py', 'native_report', {
        'IAMJB/chexpert-mimic-cxr-findings-baseline': str(root / 'models/findings'),
        'IAMJB/chexpert-mimic-cxr-impression-baseline': str(root / 'models/impression')
    }).ChestXRayReportGeneratorTool(cache_dir=str(root / 'cache/huggingface'), device='cuda:0')
    vqa_cls = load_module(tools_dir / 'xray_vqa.py', 'native_vqa').XRayVQATool
    from transformers import dynamic_module_utils
    from transformers.cache_utils import DynamicCache
    # CheXagent's TensorFlow import is TYPE_CHECKING-only; HF's scanner treats it as runtime.
    original_imports = dynamic_module_utils.get_imports
    def runtime_imports(filename):
        imports = original_imports(filename)
        if Path(filename).name == 'tokenization_chexagent.py':
            imports = [name for name in imports if name != 'tensorflow']
        return imports
    # The pinned model uses the 4.40 names for these unchanged DynamicCache operations.
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(lambda cache: cache.get_seq_length())
    if not hasattr(DynamicCache, 'get_max_length'):
        DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
    version = transformers.__version__
    # Keep downloaded assets intact; the small runtime config resolves the nested visual model locally.
    runtime_assets = root / 'models/chexagent_runtime'
    runtime_assets.mkdir(exist_ok=True)
    for asset in (root / 'models/chexagent').iterdir():
        target = runtime_assets / asset.name
        if asset.name == 'config.json':
            config = json.loads(asset.read_text())
            config['visual']['vision_model_name_or_path'] = str(root / 'models/xraysiglip')
            target.write_text(json.dumps(config, indent=2) + '\n')
        elif not target.exists():
            target.symlink_to(asset)
    try:
        dynamic_module_utils.get_imports = runtime_imports
        expert = vqa_cls(model_name=str(runtime_assets), device='cuda:0',
                         cache_dir=str(root / 'cache/huggingface'))
    finally:
        transformers.__version__ = version
        dynamic_module_utils.get_imports = original_imports
    # Count expert/report generations separately from planner calls, with actual token lengths.
    for label, model in [('expert', expert.model), ('findings', report.findings_model),
                         ('impression', report.impression_model)]:
        original = model.generate
        def measured(*args, _generate=original, _label=label, **kwargs):
            guard()
            start = time.time()
            record('expert_generation_started', {'model': _label})
            with torch.inference_mode():
                value = _generate(*args, **kwargs)
            input_length = args[0].shape[-1] if _label == 'expert' else 0
            record('expert_generation_finished', {'model': _label, 'seconds': time.time() - start,
                'generated_sequence_length': int(value.shape[-1] - input_length)})
            return value
        model.generate = measured
    state = {'calls': 0, 'max_calls': 8, 'images': set()}
    tools = [BoundedTool(t, state, record, guard) for t in [classifier, segment, expert, report]]
    agent = load_module(root / 'upstream/medrax/agent/agent.py', 'native_agent').Agent
    prompt = (root / 'upstream/medrax/docs/system_prompts.txt').read_text().split('[MEDICAL_ASSISTANT]\n', 1)[1].split('\n[GENERAL_ASSISTANT]', 1)[0].strip()
    return tools, state, agent, prompt


class NoUpdateHCL:
    """Disabled intervention baseline: no prompt, memory, routing or state mutation."""
    def __init__(self, model):
        self.model = model

    def bind_tools(self, tools):
        self.model.bind_tools(tools)
        return self

    def invoke(self, messages):
        return self.model.invoke(messages)
