import json
import os
from pathlib import Path
from collections import Counter
from acquire import load_vlm
from previous_learning import edit_skill, update_memory, words

ROOT = Path(os.environ["RUN_ROOT"])
design = json.loads((ROOT / "design.json").read_text())
calls = Counter()


def save(name, obj):
    (ROOT / (name + ".json")).write_text(json.dumps(obj, indent=2))


def models():
    choose, gen = load_vlm(with_text_generator=True)
    def counted(fn, kind):
        def call(*args):
            calls[kind] += 1
            if sum(calls.values()) > design["budget"]["max_total_model_calls"]:
                raise RuntimeError("model budget reached")
            return fn(*args)
        return call
    return counted(choose, "decisions"), counted(gen, "generations")


def learn(seed, examples, gen):
    directory = ROOT / "memory"
    directory.mkdir()
    memories = update_memory({"memories":[]}, "fit", examples, directory, gen)
    # Bounded balanced evidence for the editor; native memory sees all fit examples.
    right = [x for x in examples if x["response"] == x["answer"]]
    wrong = [x for x in examples if x["response"] != x["answer"]]
    inputs = wrong[:12] + right[:8]
    skill, attempts = edit_skill(seed, inputs, memories, gen)
    save("private_update", {"attempts":attempts,"skill":skill,"memory":memories})
    return skill, memories[0]


def guide(skill, memory, query):
    def similarity(row):
        a, b = words(query), words(row["question"])
        return len(a & b) / max(1, len(a | b))
    row = max(memory["raw"], key=similarity)
    abstract = " ".join(a["guidance"] for a in memory["abstract"] if a.get("guidance"))
    return (skill + "\nPrior fit example: " + row["question"][:450] + "\nPreferred action: "
            + str(row["correct_feedback"])[:160] + "\nExperience: " + abstract[:400])


def example(k, context, observed, target):
    return dict(task_id=str(k), question=context, response=observed, answer=target,
                split="fit", task_type="tool_selection")
