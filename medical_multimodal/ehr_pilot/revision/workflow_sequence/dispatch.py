"""Question-only intent selection and composed workflow invocation."""

import os
import re
import sys
from collections import Counter
from urllib.parse import parse_qs, urlsplit

import workflow as w

b, p, d, old = w.b, w.p, w.d, w.old
CLASSES = dict(A=4, B=5, C=6, D=7, E=9, F=10, G=0)
TASKS = (
    "A. Return the latest magnesium result within a specified window.\n"
    "B. Take a conditional magnesium replacement action.\n"
    "C. Return an average result without a replacement action.\n"
    "D. Return the latest glucose result.\n"
    "E. Take a conditional potassium replacement and schedule its recheck.\n"
    "F. Report hemoglobin A1c and conditionally order a recheck.\n"
    "G. Take a conditional potassium replacement and recheck, and also return the potassium mean.\n"
)


def question(case, composite):
    text = case["instruction"]
    if not composite:
        return text
    return text + " Also report the mean of all potassium values in the 24 hours ending at the current time."


def candidates():
    eicu = b.datasets()["eicu"]["valid"]
    return [("native", c, False) for c in w.fresh_cases()] + [("compose", c, True) for c, _ in eicu]


def infer():
    v = w.v
    v.setup_model()
    part = int(os.environ["PART"])
    output = []
    queue = candidates()[part::2]
    token_ids = [p.tok.encode(x, add_special_tokens=False) for x in CLASSES]
    assert all(len(x) == 1 for x in token_ids)
    indices = [x[0] for x in token_ids]
    prefix = "Identify the required workflow from the question. Select one letter.\n" + TASKS
    with p.torch.inference_mode():
        for start in range(0, len(queue), 4):
            chunk = queue[start:start+4]
            prompts = [prefix + "\nQuestion: " + question(case, comp) for _, case, comp in chunk]
            texts = [p.tok.apply_chat_template([dict(role="user", content=x)], tokenize=False,
                                               add_generation_prompt=True) for x in prompts]
            enc = p.tok(texts, return_tensors="pt", padding=True).to("cuda")
            ranks = p.model(**enc, use_cache=False).logits[:, -1, indices].argmax(dim=1).tolist()
            generated = p.generate([[dict(role="user", content=(
                "Generate only the laboratory query: GET /fhir/Observation?patient=<identifier>&code=<abbreviation>. "
                "Use MG for magnesium, K for potassium, GLU for glucose, A1C for hemoglobin A1c.\n"
                + question(case, composite)
            ))] for _, case, composite in chunk], limit=80)
            for (panel, case, _), idx, action in zip(chunk, ranks, generated, strict=True):
                output.append(dict(panel=panel, id=case["id"], kind=list(CLASSES.values())[idx], action=action.strip()))
            p.save(f"dispatch_{part}.json", output)
            print(part, len(output), "/", len(queue), flush=True)


def valid_query(case, text):
    match = re.fullmatch(r"GET\s+(\S+)", text.strip().strip("`"))
    if not match:
        return False
    raw = match[1]
    if raw.startswith("/"):
        address = urlsplit(raw)
    elif raw.startswith(p.BASE):
        address = urlsplit(raw)
    else:
        return False
    target = urlsplit(d.query(case))
    return address.path == target.path and parse_qs(address.query) == parse_qs(target.query)


def report():
    p.ref = p.scorer()
    examples, panels = w.bank_rows()
    model = w.fit(examples)
    previous = {r["id"]: r["observation"] for shard in range(2)
                for r in old.read(f"previous/query_{shard}")}
    decisions = {(r["panel"], r["id"]): r for shard in range(2)
                 for r in old.read(f"dispatch_{shard}")}
    result = {}
    native = Counter(retrieval=0, induction=0)
    for case in w.fresh_cases():
        row = decisions.pop(("native", case["id"]))
        intent = row["kind"] == d.kind(case)
        query = valid_query(case, row["action"])
        obs = previous[case["id"]]
        native.update(n=1, scope_correct=int(intent), query_correct=int(query),
                      both_correct=int(intent and query))
        if query and obs is not None and row["kind"] != 0:
            _, truth = b.expected(case, obs)
            scoped = dict(case, id=f"task{row['kind']}_dispatch")
            for method in ("retrieval", "induction"):
                _, payload = w.execute(model, scoped, obs, method)
                metrics = b.score(case, obs, payload, truth)
                native[method] += metrics["full"]
                native[method + "_wrong_write"] += metrics["wrong_write"]
    result["native"] = native
    selected = []
    composition = Counter()
    for case, obs in panels["eicu"]["valid"]:
        row = decisions.pop(("compose", case["id"]))
        intent = row["kind"] == 0
        query = valid_query(case, row["action"])
        composition.update(n=1, scope_correct=int(intent), query_correct=int(query),
                           both_correct=int(intent and query))
        if intent and query:
            selected.append((case, obs))
    assert not decisions
    result["compose"] = composition
    result["compose_completed_among_dispatched"] = w.compose(selected, model)
    p.save("dispatch_result.json", result)


if __name__ == "__main__":
    {"infer": infer, "report": report}[sys.argv[1]]()
