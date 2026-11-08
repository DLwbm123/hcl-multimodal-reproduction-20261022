"""Development integration policies; learned applicability is not claimed."""
import json
import re

ARMS = ["H0", "Raw", "Fixed", "Proposed_v0"]
FOCUSED = "Does this chest X-ray show airspace opacity or consolidation? Answer yes or no."
FINAL = "If you had to choose the best option, only respond with the letter of choice (only one of A, B, C, D, E, F)"
STOP = set("the and for with from this that these those which what based provided following most likely correct answer option image images imaging chest xray radiograph radiographs findings patient figure".split())


def stem(text):
    match = re.search(r"(?m)^\s*[A-F][.)]\s+", text)
    return (text[:match.start()] if match else text).strip()


def option(question, label):
    matches = list(re.finditer(r"(?m)^\s*([A-F])[.)]\s+", question))
    choices = {m[1]: question[m.end():matches[i+1].start() if i+1 < len(matches) else len(question)].strip()
               for i, m in enumerate(matches)}
    return choices.get(label, "Unparseable historical option; do not infer an answer")


def raw_context(bank, question):
    terms = lambda s: {x for x in re.findall('[a-z0-9]+', stem(s).lower()) if len(x) > 2 and x not in STOP}
    q = terms(question)
    def rank(row):
        t = terms(row['question'])
        return (-(len(q & t) / len(q | t) if q | t else 0), row['memory_id'])
    chosen = sorted(bank, key=rank)[:2]
    payload = [{'historical_question_stem': stem(x['question'])[:1500],
                'historical_attempt_option_text': option(x['question'], x['response'])[:1500],
                'unadjudicated_benchmark_target_text': option(x['question'], x['feedback'])[:1500]}
               for x in chosen]
    warning = ('\nTwo historical training examples from DIFFERENT cases. Their benchmark feedback was NOT independently '
               'clinically adjudicated and can be ambiguous or wrong. Use them as conditional past experience, never as facts '
               'or diagnoses about the current patient. Do not copy an answer, invent absent findings, or infer chronology '
               'or causation from historical supervision. Current images and the original question remain primary.\n')
    return warning + json.dumps(payload), [x['memory_id'] for x in chosen]


COMMON = ('\nTool execution contract: chest_xray_expert must receive one currently attached image per request, '
          'with max_new_tokens at most 256. Multi-image inputs may exceed its context window: if needed, plan '
          'separate image-bound requests within the same total budget. A failed or rejected tool call supplies no '
          'positive or negative medical evidence. Do not infer time ordering from anonymized filenames. '
          'Image arrows and original figure labels may be present and are part of the supplied input.\n')

FIXED = ('\nFixed, non-updating observation workflow: If there is exactly one supplied image, and distinguishing '
         'whether airspace opacity or consolidation is present is relevant to the current question, obtain that '
         'observation using chest_xray_expert with this exact prompt: ' + FOCUSED + '\n'
         'Skip the additional request if an explicit, non-conflicting observation of that same target is already '
         'available from a successful tool call on the current image. Do not use this narrow query for a nodule, '
         'pleural mass, interstitial abnormality, chronology, causal diagnosis or severity as if they were the target. '
         'If the task is not applicable, multiple images are supplied, or evidence is uncertain, continue the original '
         'Agent workflow. Use the returned observation only as evidence about the specified target and image, '
         'never as proof of etiology or of a final answer choice. Do not repeatedly query the same target.\n')


def policy_context(arm, question, bank, candidate):
    if arm == 'H0':
        return '', []
    if arm == 'Raw':
        return raw_context(bank, question)
    if arm == 'Fixed':
        return FIXED, []
    if arm != 'Proposed_v0':
        raise ValueError('Unknown arm')
    if candidate['action']['prompt'] != FOCUSED or candidate['status'] != 'qualified_for_agent_development_only':
        raise ValueError('Candidate differs from the qualified source operation')
    # Fields originate in the existing feedback-selected artifact. Semantic dispatch is
    # performed by the actual controller; no current answer or prefilled predicate is used.
    entry = {k: candidate[k] for k in ['predicate', 'modality', 'image_count', 'output_type', 'action']}
    return ('\nA development-only operation was selected by historical paired feedback. It is not clinical knowledge. '
            'Candidate: ' + json.dumps(entry) + '\n'
            'Decide from the current question, attached images and actual tool observations whether this operation '
            'addresses a missing fact that matters to the answer. Select it only when there is exactly one attached '
            'image and the missing fact is the presence of airspace opacity or consolidation. The current question '
            'may use that observation as an intermediate fact; it need not itself be yes/no. '
            'If there is no such information gap, if the target is a different kind of opacity, or if an explicit '
            'non-conflicting successful tool observation already supplies that fact, reject the candidate and follow '
            'the original workflow. Multiple images are outside this candidate\'s qualified scope. '
            'When selected, call the named existing tool with the exact candidate prompt and one current image; '
            'never invent an observation. Do not repeatedly query the same target. Associate the result with this '
            'image and this call; uncertainty or a failed execution is not absence. The observation does not establish '
            'etiology, chronology, severity, or the correct answer choice.\n'), []


def summarize(inputs, episodes, gold):
    keys = {(x['id'], arm) for x in inputs for arm in ARMS}
    seen = [(x['id'], x['arm']) for x in episodes]
    if len(seen) != len(set(seen)) or not set(seen) <= keys:
        raise ValueError('Duplicate or out-of-manifest episode')
    by = {(x['id'], x['arm']): x for x in episodes}
    metric = {}
    for arm in ARMS:
        rows = [by.get((x['id'], arm), {}) for x in inputs]
        metric[arm] = {'denominator': len(inputs),
                       'completed': sum(x.get('status') == 'completed' for x in rows),
                       'correct': sum(x.get('predicted') == gold[item['id']] for item, x in zip(inputs, rows)),
                       'controller_calls': sum(x.get('controller_calls', 0) for x in rows),
                       'tool_calls': sum(x.get('tool_calls', 0) for x in rows),
                       'seconds': sum(x.get('seconds', 0) for x in rows),
                       'focused_requests': sum(x.get('focused_requests', 0) for x in rows)}
    def pairs(a, b, subset):
        result = {'denominator': len(subset), 'gain': 0, 'loss': 0}
        for item in subset:
            left = by.get((item['id'], a), {}).get('predicted') == gold[item['id']]
            right = by.get((item['id'], b), {}).get('predicted') == gold[item['id']]
            result['gain'] += int(not left and right)
            result['loss'] += int(left and not right)
        return result
    subsets = {'all_benchmark_targets': inputs,
               'source_aligned_not_adjudicated': [x for x in inputs if x['reference_tier'].startswith('source_aligned')],
               'unresolved_or_conflicting_reference': [x for x in inputs if not x['reference_tier'].startswith('source_aligned')]}
    return {'planned_episodes': len(keys), 'recorded_episodes': len(episodes), 'metrics': metric,
            'paired': {s: {a + '_to_Proposed_v0': pairs(a, 'Proposed_v0', rows) for a in ['H0', 'Raw', 'Fixed']}
                       for s, rows in subsets.items()},
            'confirmation_used': False, 'clinical_correctness_established': False,
            'learned_conditional_boundary_established': False, 'automatic_promotion_allowed': False,
            'purpose': 'real-controller development integration; benchmark concordance, not clinical validation'}
