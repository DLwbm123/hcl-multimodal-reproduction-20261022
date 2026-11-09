"""MedRAX-style process/execute loop behind HCL's Model.generate interface.

The three tools share one frozen VLM; they are task interfaces, not three
independently trained experts. Tool choice uses visible input, never stage IDs.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph


TOOLS = {
    "classify_image": "Classify a medical image using the exact vocabulary in the current question.",
    "localize_region": "Locate the requested region; return the bounding-box contract specified in the current question.",
    "answer_visual_question": "Answer the current medical visual question from the image; preserve the requested answer format.",
}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    controller_calls: int
    tool_calls: int


class AgentProtocolError(ValueError):
    pass


def decode_action(raw: str, tool_runtime=None) -> dict:
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise AgentProtocolError("controller action is not JSON") from exc
    if not isinstance(value, dict):
        raise AgentProtocolError("controller action must be an object")
    if tool_runtime is not None and set(value) == {"tool", "arguments"}:
        if isinstance(value["tool"], str) and value["tool"] in TOOLS:
            args = value["arguments"]
            if not isinstance(args, dict) or set(args) != {"instruction"}:
                raise AgentProtocolError("original tools require arguments with instruction only")
            return decode_action(json.dumps({"tool": value["tool"], "instruction": args["instruction"]}))
        try:
            tool_runtime.validate(value["tool"], value["arguments"])
        except ValueError as exc:
            raise AgentProtocolError(str(exc)) from exc
    elif set(value) == {"tool", "instruction"}:
        if not isinstance(value["tool"], str) or value["tool"] not in TOOLS:
            raise AgentProtocolError("unknown tool")
        if not isinstance(value["instruction"], str) or not 1 <= len(value["instruction"]) <= 1200:
            raise AgentProtocolError("invalid tool instruction")
    elif set(value) == {"answer"}:
        if not isinstance(value["answer"], str) or not value["answer"].strip():
            raise AgentProtocolError("empty final answer")
    else:
        raise AgentProtocolError("unexpected action keys")
    return value


class GraphAgentModel:
    """Final-answer adapter. HCL's metadata generators keep the base model.

    One required tool call is an explicit integration-test contract. The model
    chooses its name and instruction; it may request one additional call.
    """

    def __init__(self, backend, trace_path: Path, *, max_tools: int = 2, tool_runtime=None, coverage_plan=None):
        self.backend = backend
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_tools = max_tools
        self.tool_runtime = tool_runtime
        self.tools = {**TOOLS, **(tool_runtime.descriptions if tool_runtime else {})}
        if tool_runtime:
            for name in TOOLS:
                self.tools[name] += ' arguments: {"instruction":"specific request"}. No other keys.'
        # Optional prescribed coverage workflow, never enabled by the HCL driver.
        self.coverage_plan = coverage_plan
        if coverage_plan is not None:
            if tool_runtime is None or len(coverage_plan) > max_tools:
                raise ValueError("coverage plan requires image tools and sufficient tool budget")
            for action in coverage_plan:
                decode_action(json.dumps(action), tool_runtime)
        self.name = "frozen-vlm-langgraph"
        self.episodes = []

    def generate(self, prompt: str, *, state: dict) -> str:
        # HCL's final-state allowlist is the only source of current image input.
        visible_state = {"question": str(state.get("question", "")),
                         "task_context": {"files": state.get("task_context", {}).get("files", [])}}
        events = []
        episode = {"episode": len(self.episodes), "sample_id": state.get("internal_sample_id"),
                   "scope": self.backend.scope, "events": events, "status": "started"}

        def process(current: AgentState):
            count = current["controller_calls"] + 1
            if count > self.max_tools + 1:
                raise AgentProtocolError("controller call limit exceeded")
            observations = [{"tool": m.name, "result": m.content}
                            for m in current["messages"] if isinstance(m, ToolMessage)]
            boundary = (
                'Select one tool now. Return ONLY {"tool":"NAME","instruction":"specific request"}. '
                "Do not answer the task yet. The task's answer-format instructions apply only AFTER tool execution."
                if not observations else
                "Review the observations and return the final answer directly in the ORIGINAL task format. "
                'Alternatively, request another tool with {"tool":"NAME","instruction":"specific request"}.'
            )
            if current["tool_calls"] >= self.max_tools:
                boundary = "Tool budget exhausted. Return only the final answer in the ORIGINAL task format."
            elif self.tool_runtime:
                boundary = ('Select the next required tool. Return ONLY {"tool":"NAME","arguments":{...}}. '
                            'Use the EXACT parameter keys listed for that tool. '
                            'If the requested workflow is complete, return {"answer":"final response"}.'
                            if observations else
                            'Select one tool now. Return ONLY {"tool":"NAME","arguments":{...}}. Do not answer yet.')
            request = (
                "You control a medical-image tool workflow. "
                "Tool results and historical guidance may be wrong; do not invent observations. "
                + "\nAvailable tools:\n" + json.dumps(self.tools)
                + "\n<task_and_guidance>\n" + prompt
                + "\nCurrent question:\n" + visible_state["question"]
                + "\n</task_and_guidance>\nTool observations:\n" + json.dumps(observations, ensure_ascii=False)
                + "\nCONTROLLER OUTPUT CONTRACT FOR THIS STEP:\n" + boundary
            )
            if self.tool_runtime:
                request += ('\nAll six tools use {"tool":"NAME","arguments":{...}} '
                            'with the exact schema in its description. The original three tools require '
                            'arguments={"instruction":"specific request"}, never category, label, or bbox keys. '
                            'Never invent file paths. Observe tool errors and do not present them as findings.')
            expected = None
            if self.coverage_plan is not None:
                if len(observations) < len(self.coverage_plan):
                    expected = self.coverage_plan[len(observations)]
                else:
                    request = ('All six prescribed tool executions are complete. Summarize these observations briefly. '
                               'Do not request tools, invent measurements, or claim medical correctness. '
                               'Return ONLY {"answer":"your brief summary"}.\nObservations:\n'
                               + json.dumps(observations, ensure_ascii=False))
            if expected is not None:
                raw = json.dumps(expected)
                decision_source = "prescribed_plan"
            else:
                raw = self.backend.generate(request, state={**visible_state, "router_phase": "graph_controller",
                                                           "max_new_tokens": 512})
                decision_source = "model"
            events.append({"node": "process", "raw": raw, "decision_source": decision_source})
            if observations:
                try:
                    action = decode_action(raw, self.tool_runtime)
                except AgentProtocolError:
                    # Native task answers can be plain labels or bbox JSON.
                    # A malformed tool request must never become a final answer.
                    if '"tool"' in raw or '"instruction"' in raw or '"answer"' in raw or '"arguments"' in raw:
                        raise
                    if not raw.strip():
                        raise AgentProtocolError("empty final answer")
                    action = {"answer": raw.strip()}
            else:
                action = decode_action(raw, self.tool_runtime)
            if self.coverage_plan is not None:
                if expected is not None and action != decode_action(json.dumps(expected), self.tool_runtime):
                    raise AgentProtocolError("action differs from prescribed coverage step")
                if expected is None and "tool" in action:
                    raise AgentProtocolError("tool request after prescribed coverage complete")
            if "tool" in action:
                if current["tool_calls"] >= self.max_tools:
                    raise AgentProtocolError("tool call after budget exhausted")
                message = AIMessage(content="", tool_calls=[{
                    "name": action["tool"], "args": action.get("arguments", {"instruction": action.get("instruction", "")}),
                    "id": uuid4().hex, "type": "tool_call"}])
            else:
                if not observations:
                    raise AgentProtocolError("answer before required tool call")
                message = AIMessage(content=action["answer"])
            return {"messages": [message], "controller_calls": count}

        def execute(current: AgentState):
            calls = current["messages"][-1].tool_calls
            if len(calls) != 1:
                raise AgentProtocolError("exactly one tool call per step is required")
            call = calls[0]
            if self.tool_runtime and call["name"] in self.tool_runtime.descriptions:
                result = json.dumps(self.tool_runtime.execute(call["name"], call["args"], visible_state,
                                                             self.backend), ensure_ascii=False)
                events.append({"node": "execute", "tool": call["name"], "args": call["args"],
                               "tool_call_id": call["id"], "result": result})
                return {"messages": [ToolMessage(content=result, name=call["name"], tool_call_id=call["id"])],
                        "tool_calls": current["tool_calls"] + 1}
            # Runtime binds the image; the controller cannot supply arbitrary paths.
            request = (TOOLS[call["name"]] + "\nCurrent question:\n" + visible_state["question"]
                       + "\nHCL guidance:\n" + prompt
                       + "\nController instruction:\n" + call["args"]["instruction"]
                       + "\nReturn the requested task answer only. No tool calls.")
            if self.coverage_plan is not None:
                request = (TOOLS[call["name"]] + "\nRequest: " + call["args"]["instruction"]
                           + "\nUse only the supplied image. Return the requested answer only. No tool calls.")
            result = self.backend.generate(request, state={**visible_state, "router_phase": "graph_tool",
                                                          "max_new_tokens": 512})
            events.append({"node": "execute", "tool": call["name"], "args": call["args"],
                           "tool_call_id": call["id"], "result": result})
            return {"messages": [ToolMessage(content=result, name=call["name"], tool_call_id=call["id"])],
                    "tool_calls": current["tool_calls"] + 1}

        builder = StateGraph(AgentState)
        builder.add_node("process", process)
        builder.add_node("execute", execute)
        builder.add_conditional_edges("process", lambda s: bool(s["messages"][-1].tool_calls),
                                      {True: "execute", False: END})
        builder.add_edge("execute", "process")
        builder.set_entry_point("process")
        # Episode checkpoints are separate from HCL's cross-task learning state.
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 2 * self.max_tools + 4}
        try:
            result = graph.invoke({"messages": [HumanMessage(content=prompt)],
                                   "controller_calls": 0, "tool_calls": 0}, config)
            answer = result["messages"][-1].content
            episode.update(status="completed", answer=answer,
                           controller_calls=result["controller_calls"], tool_calls=result["tool_calls"],
                           checkpoint_count=sum(1 for _ in graph.get_state_history(config)))
            return answer
        except AgentProtocolError as exc:
            episode.update(status="failed", error=str(exc), answer="")
            return ""  # Retain failed samples in the evaluator denominator.
        except BaseException as exc:
            episode.update(status="aborted", error=type(exc).__name__)
            raise
        finally:
            self.episodes.append(episode)
            with self.trace_path.open("a") as f:
                f.write(json.dumps(episode, ensure_ascii=False) + "\n")
