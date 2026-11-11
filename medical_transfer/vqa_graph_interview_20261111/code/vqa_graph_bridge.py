"""Attach the existing bounded graph without resetting a learned HCL harness."""
from types import MethodType
from graph_agent import GraphAgentModel


def attach_graph(pipeline, backend, trace_path, *, max_tools=2, tool_runtime=None, max_action_retries=1):
    if isinstance(pipeline.model, GraphAgentModel):
        raise ValueError('graph already attached')
    if pipeline.read_only_workers != 1 or pipeline.prediction_cache.enabled:
        raise ValueError('graph adapter requires serial uncached execution')
    if pipeline.router_config.use_chat_tool_calls or pipeline.router_config.final_generation_cache_path:
        raise ValueError('disable alternate tool path and final-generation cache first')
    if not 1 <= max_tools <= 6:
        raise ValueError('tool budget must be between one and six')
    graph = GraphAgentModel(backend, trace_path, max_tools=max_tools, tool_runtime=tool_runtime, max_action_retries=max_action_retries)
    # Metadata, memory consolidation and candidate generation retain their existing backends.
    pipeline.model = graph
    original_batch = pipeline._evaluate_train_batch

    def with_graph_feedback(self, *args, **kwargs):
        first = len(graph.episodes)
        result = original_batch(*args, **kwargs)
        episodes = graph.episodes[first:]
        counts = {}
        for episode in episodes:
            for event in episode['events']:
                if event['node'] == 'execute':
                    counts[event['tool']] = counts.get(event['tool'], 0) + 1
        result['metrics']['graph_execution'] = {
            'episodes': len(episodes),
            'completed': sum(e['status'] == 'completed' for e in episodes),
            'tool_counts': counts,
            'failures': [e.get('error', 'aborted') for e in episodes if e['status'] != 'completed'],
        }
        return result

    pipeline._evaluate_train_batch = MethodType(with_graph_feedback, pipeline)
    return graph
