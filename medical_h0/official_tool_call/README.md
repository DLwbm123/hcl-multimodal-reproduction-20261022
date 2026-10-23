# Official Qwen function calling for restricted MedTool

This change replaces the bespoke whole-response JSON protocol with **Qwen-Agent's `BaseFnCallModel`**, configured with `fncall_prompt_type="nous"` as in the Qwen2.5-VL official computer-use cookbook. Tool descriptions, function-call serialization, tool-result formatting, assistant continuation, stop-word postprocessing and call extraction use upstream code. The local subclass supplies only bounded Transformers generation. The LangChain bridge validates tool-result IDs/names and JSON Schema before the unchanged restricted dispatcher runs.

The checkpoint chat template alone does not render tool schemas. Passing `tools=` to that template is not an equivalent integration. The official layer still uses learned text markers (`<tool_call>`, `<tool_response>`); this is not constrained decoding and cannot guarantee valid or semantically correct arguments. We do not repair JSON, add missing values, or execute unregistered tools. The upstream Nous parser accepts JSON5 and can extract incomplete calls; our bridge additionally requires object arguments and registered JSON Schema before dispatch. Invalid requests become failed tool observations through the existing MedicalHarness graph.

Pinned upstream: Qwen-Agent commit `31a4d36d123688581a9e9744427272b33ce940e0` (package version 0.0.34), and MedicalHarness `8727186f04c25646e7ee1b9428f3b6dcd4239382`. Sources:

- [Official function-calling guide](https://qwen.readthedocs.io/en/v2.5/framework/function_call.html)
- [Legacy Qwen2-VL compatibility example](https://github.com/QwenLM/Qwen-Agent/blob/31a4d36d123688581a9e9744427272b33ce940e0/examples/qwen2vl_function_calling.py)
- [Official function-call implementation](https://github.com/QwenLM/Qwen-Agent/blob/31a4d36d123688581a9e9744427272b33ce940e0/qwen_agent/llm/function_calling.py)

- [Qwen2.5-VL official cookbook](https://github.com/QwenLM/Qwen2.5-VL/blob/3c4f939dda856110cefb315df6280b9b4eb15219/cookbooks/computer_use.ipynb) explicitly uses `NousFnCallPrompt` with Qwen2.5-VL-7B. The initial legacy-protocol run was retained as a failed engineering attempt; both attempts share the original clock and call/storage caps.

Run CPU checks after installing upstream dependencies:

```sh
python -m unittest discover -s . -p 'test_*.py' -v
python acceptance.py --mode scripted --output OUTPUT --medicalharness MEDICALHARNESS_CHECKOUT
```

Real mode additionally needs the existing HCL model loader path, local Qwen2.5-VL weights, a fresh `budget.json` and a live independent guard writing `guard_status.json`. Reuse of that loader does **not** enable HCL learning. Inference stays BF16, greedy, max 1024 new tokens, context cap 16384. Only text MedTool inputs are supported by this bridge; visual acceptance is not claimed.

Acceptance uses six synthetic development episodes (two prior failures, two new arithmetic inputs, two injected first-calculation failures). Frozen gate: all six submit correct results through real facts/calculation calls, both injected failures recover, zero invalid parsed/schema calls, and all 13 boundary/protocol CPU tests pass. This is separate from clinical H0 and continuing-learning acceptance. The previous 96-call failed protocol remains closed and preserved.

The executor exposes only current-task fact reads, restricted arithmetic, submission and the official per-episode planner. It is an application capability boundary, **not an OS sandbox**. No built-in Qwen-Agent tools, shell, browser or code interpreter are enabled.
