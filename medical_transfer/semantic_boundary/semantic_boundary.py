"""One opt-in final-context change; retrieval and memory records stay unchanged."""
from types import MethodType

MEMORY_BOUNDARY = """
Historical-memory evidence boundary:
Every selected_memory record describes a different historical case/image. Its
correct_feedback is authoritative only for that historical case; previous_attempt
is a historical attempt and may be wrong. Neither field, nor the selector's reason,
is evidence of the findings, absence of findings, count, class or coordinates in
the currently supplied image. Do not transfer a historical answer to this image.
You may use a relevant general procedure or output-format lesson from memory.
Determine this case's visual facts from the current image and visible question,
then obey the current output contract. Return only the requested answer.
""".strip()

def bounded_context(context, selected_memory):
    return context + '\n\n' + MEMORY_BOUNDARY if selected_memory else context

def install_semantic_boundary(pipeline):
    if getattr(pipeline, '_semantic_boundary_installed', False):
        raise ValueError('already installed')
    if pipeline.harness_version or pipeline.global_batch_index or pipeline.checkpoint.has_state:
        raise ValueError('fresh pipeline required')
    original = pipeline._predict_with_router
    def predict(self, examples, router, **kwargs):
        original_build = router.build_context
        def build(router_self, *args, **build_kwargs):
            context = original_build(*args, **build_kwargs)
            return bounded_context(context, build_kwargs.get('selected_memory', []))
        router.build_context = MethodType(build, router)
        try:
            return original(examples, router, **kwargs)
        finally:
            router.build_context = original_build
    pipeline._predict_with_router = MethodType(predict, pipeline)
    pipeline._semantic_boundary_installed = True
    return pipeline
