# Learned VQA memory through a bounded LangGraph agent

This integration loads the committed memory from the completed VQA experiment and routes final answering through the existing LangGraph controller/tool loop. HCL's input structuring, memory selection, judge, candidate-generation and memory backends remain unchanged. The learned state is restored before graph attachment. No new efficacy score or training claim is made: the preceding 74/200 result used the direct VLM path.

Three predetermined, already-exposed training examples were used, with six tools available, at most two tool executions per episode and no prescribed tool plan. The real controller chose `answer_visual_question` twice in every successful episode; this does not demonstrate use of all six tools or optimal tool selection. Selected historical-memory counts were 2, 2 and 1. Every evaluation preserved the original 10 raw records, two abstracts, 72 evaluator-only anchors and zero skills, with frozen model parameters.

The first pass completed 2/3 cases. One action had empty arguments, violating the required `instruction` schema. Version 2 adds **one opt-in action-validation retry**, preserving the original examples, clock and cumulative calls. The rejected action is never executed or silently repaired: its validation error returns to the controller, which must issue a corrected action. The same three cases then completed 3/3. One real case followed:

`process → execute → process → validation_error → process → execute → process`

The other cases followed two tool rounds without errors. Graph checkpoint counts were 7, 8 and 7; controller calls were 3, 4 and 3. Both passes used 47 cumulative model calls (22 + 25) and 254.12 seconds from the original start. Supervisor and worker exited normally. Three new CPU contract tests and all seven existing integration regressions passed using scripted model fixtures.

## Implementation

- `code/vqa_graph_bridge.py`: replace only the final-answer backend and append training-episode process diagnostics when the original training method is invoked. The smoke run itself is read-only.
- `../graph_integration/graph_agent.py`: LangChain `HumanMessage`/`AIMessage`/`ToolMessage`, LangGraph `StateGraph`, conditional edges and in-memory episode checkpoints. The new retry parameter defaults to zero, preserving existing entry points; this VQA bridge enables one retry. Tool/controller/recursion budgets remain bounded.
- `code/worker.py`: restore the committed HCL memory, attach the graph, run three fixed inputs, verify state equality and frozen weights, save private traces and public receipts.
- `code/supervisor.py`: independent 15-minute hard timeout, 1 GB early storage stop and neutral process-entry paths supplied by environment.
- `code/check_graph.py`: graph/message flow, label isolation, invalid-action recovery, duplicate attachment and repeated-tool-call bounds.

Set `RUN_ROOT`, `WORKER_PYTHON`, `WORKER_ENTRY` and `FORBIDDEN_PROCESS_TERMS` using the existing deployment workflow. Fill the private-resource placeholders in `protocol.template.json`. The worker requires the preceding VQA adapter module plus the existing coordinate, semantic, task-scope, medical-bridge and HCL modules; the graph and image-tool runtime come from sibling `graph_integration` and `six_tool_demo` directories. All versions used were already installed: LangGraph 1.2.11, langchain-core 1.6.2, Torch 2.7.1+cu128, Transformers 4.51.3 and OpenCV 4.11.0.

`results.json` retains both attempts as aggregate evidence. Private images, questions, answers, memory contents, paths and individual traces are omitted, so this is not a self-contained private-state reproduction bundle. The original confirmation set was not reused, and the 74/200 result must not be attributed to this graph.

The implementation uses project-validated JSON actions converted into LangChain tool-call messages, not provider-native `bind_tools`, `ToolNode` or `create_agent`. `InMemorySaver` records one episode and does not provide cross-process durable recovery. See [the interview explanation](INTERVIEW.zh-CN.md) for the precise distinctions and code reading order.
