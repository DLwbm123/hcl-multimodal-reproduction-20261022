"""A fixed, generic evidence-checking instruction; no new tools or oracle checks."""
import bootstrap_v8
from adapter_v7 import adapter_class as previous_adapter
from langchain_core.messages import SystemMessage

GUIDANCE = '''Before executing a calculation or ending a task, check the evidence against each requested requirement:
1. Match both the target patient and target episode. Other-patient or other-episode observations are distractors, not contradictions within the target. Same-scope facts on unrelated fields are not evidence for the requested field.
2. For each required field, distinguish absent, consistently supported, and conflicting evidence. If incompatible values for that required field coexist in the target scope and no precedence rule is supplied, explicitly abstain and name the conflict. Reporting both conflicting IDs as success does not resolve the conflict. Consistent repeated records are not conflicts; report all matching IDs when the task requires all.
3. For every calculator parameter, identify whether it is copied directly, normalized as text, converted in units, or derived from several quantities. A derived parameter must cite the source fact for EVERY operand used, not only one of them. State the actual arithmetic or unit conversion in transformation. A string-normalization label does not explain a numeric derivation. Do not use an unrelated fact to complete a citation list.
4. Recheck that each parameter value is supported by its cited facts and that its explanation describes that operation. If a required input is missing or has unresolved same-scope conflicts, abstain with that specific reason. A successful tool response alone does not establish evidential support. Never invent missing inputs or a priority rule.
Use the existing tool schemas exactly; this checklist supplies no facts, answers or replacement results.'''


def policy_system(base, arm):
    if arm not in ('R','I'):
        raise ValueError('Unknown policy arm')
    return base + ('\n\n'+GUIDANCE if arm=='I' else '')


def adapter_class(upstream):
    class EvidenceHarness(previous_adapter(upstream)):
        def __init__(self,*args,case,**kwargs):
            # Both policies receive the identical full-raw-input condition.
            super().__init__(*args,case={**case,'arm':'B'},**kwargs)
            self.preamble[0] = SystemMessage(content=policy_system(self.preamble[0].content,case['arm']))
            self.preamble_chars = sum(len(m.content) for m in self.preamble)
            self.manifest.update(protocol='evidence_guidance_v8',input_arm='B',policy_arm=case['arm'])
    return EvidenceHarness
