import json
import os
import re
from pathlib import Path

from hcl.memory.abstract import AbstractMemory
from hcl.memory.raw import RawMemory
from patch_format import normalize
from upstream_skill import apply_patch_with_report

DESIGN = json.loads((Path(os.environ["RUN_ROOT"]) / "design.json").read_text())


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def update_memory(st, stage, examples, directory, generator, *, capacity=4):
    raw = RawMemory(directory / "raw.jsonl", capacity_per_task=capacity)
    raw.add(stage,examples,[dict(task_id=x["task_id"],answer=x["response"]) for x in examples])
    class Adapter:
        def generate(self,prompt,*,state):
            self.output = generator(prompt,state["max_new_tokens"])
            return self.output
    adapter = Adapter()
    abstract = AbstractMemory(directory / "abstract.json",model=adapter,mode="llm",max_new_tokens=384)
    abstract.consolidate(stage,examples,raw_records=raw.records)
    (directory / "private_generated.txt").write_text(adapter.output)
    return [*st["memories"],{"raw":raw.retrieval_candidates(),"abstract":abstract.retrieval_candidates()}]


def edit_skill(current, examples, memories, gen):
    payload = [{"case_context":r["question"][:500],"executed_action":r["response"],
                "preferred_action":r["answer"]} for r in examples]
    prompt = ("Improve the reusable procedure for " + DESIGN["procedure"] + ". "
              "Reflect on recurring successes and errors in training and propose one short transferable "
              "instruction. Describe an observable condition and which available action to take. Do not identify individual training cases. Return only JSON of the form "
              '{"edits":[{"op":"append","content":"one actionable instruction"}]}. '
              'Alternatively replace the exact current instruction. Skill must stay under 700 characters. '
              "\nCurrent skill: " + current + "\nTraining trajectories: " + json.dumps(payload))
    if memories:
        prompt += "\nCandidate abstract guidance: " + json.dumps(memories[-1]["abstract"])
    attempts = []
    for _ in range(2):
        output = gen(prompt,256)
        try:
            patch, fmt = normalize(output,current)
            updated, reports = apply_patch_with_report(current,patch)
            if reports[0]["status"] not in {"applied_append","applied_replace"} or updated==current or len(updated)>700:
                raise ValueError("Unapplied, unchanged or overlong edit")
            attempts.append(dict(output=output,format=fmt,status=reports[0]["status"]))
            return updated, attempts
        except (IndexError,KeyError,ValueError,TypeError) as exc:
            attempts.append(dict(output=output,error=str(exc)))
            prompt += "\nCorrect this malformed or overlong edit: " + output + "\nError: " + str(exc)
    return None, attempts
