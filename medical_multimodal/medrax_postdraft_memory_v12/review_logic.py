"""Frozen post-draft workflow; benchmark supervision remains unadjudicated."""
import json,re
ARMS=['H0_static','generic_review','experience_review']
FIELDS=['applicability','procedure','do_not_apply','supervision_limit']
FINAL='If you had to choose the best option, only respond with the letter of choice (only one of A, B, C, D, E, F)'
INTRO='Review your initial answer before making the final choice. Your initial answer may already be correct; retain it unless the current-case evidence supports a change. Re-examine the original current-case images, question and options using the following fallible review procedures. You may use the same available tools when a specific unresolved question warrants it. Begin your checked decision with the selected option letter followed by a period, then explain it before the final letter-only turn. Any historical supervision was not clinically adjudicated; these procedures are not findings or answers for the current case.\n'
GENERIC=[{
 'applicability':'Use this check when a question asks which finding, location, characterization or diagnosis is supported by the supplied images and clinical context. Begin from the actual visible input, not from a plausible story about a similar patient.',
 'procedure':'Identify the modality and views actually present. List the decisive visible observations and their anatomical location before interpreting their cause. Compare the closest competing options explicitly, checking side, zone, extent, temporal wording, certainty and whether an option combines several claims. Check whether each decisive claim can be traced to an image or supplied sentence. Reconsider an exact dimension only when a visible scale or supplied measurement supports it. When a visual distinction is unresolved, ask an available tool a specific question about the current image and reconcile its response with the original image. Separate a tool observation from a diagnostic conclusion. Finish with the option whose full wording has the strongest support, while retaining the initial choice when review adds no contrary evidence.',
 'do_not_apply':'Do not import another patient, an absent image, a presumed earlier examination or an unprovided clinical outcome. Do not choose an option merely because it contains a familiar disease name or one correct phrase when its remaining claims lack support.',
 'supervision_limit':'This fixed generic review uses no source training examples, target labels or previous target outcomes. It is a procedural control, not expert medical adjudication.'
},{
 'applicability':'Use this check when the initial answer depends on a tool result, a subtle difference between options, a comparison over time, a measurement or an inference that goes beyond direct observation.',
 'procedure':'Name the particular uncertainty in the initial answer. Verify that every tool result belongs to the current case and addresses that uncertainty. Treat a failed call, an empty response, a bare negative, an unmatched question or repetitive output according to its actual information content; lack of useful output is not positive diagnostic evidence. Inspect whether reasoning confuses a visible finding with its etiology, overstates certainty or assumes a missing comparison. Check the complete wording of alternatives and the connection from evidence to selected option. If changing the answer, state the current-case observation or supplied fact supporting that change, rather than changing because review was requested. If uncertainty remains, preserve the distinction between observed facts and possible explanations while still selecting the best supported available option.',
 'do_not_apply':'Do not treat the request for review as proof that the first answer is wrong. Do not invent a measurement, view, history, pathology result or certainty level. Do not let an unsupported tool phrase override clearer evidence in the original input.',
 'supervision_limit':'These are fixed general checks prepared before the new run. Their length and structured fields are reported with the experience procedures; they contain no learned medical rule or current-case answer.'
}]


def parse_option(text):
    m=re.match(r'([A-F])(?:[).:\s]|$)',text.lstrip().lstrip('*'))
    return m.group(1) if m else None


def review_text(arm,states):
    if arm=='H0_static':return ''
    records=states[arm]['abstract'];assert len(records)==2
    return INTRO+json.dumps([{k:r['note'][k] for k in FIELDS} for r in records],ensure_ascii=False)


def decision(pairs,by,ids,gold,complete):
    repaired=[]
    for k in ids:
        good=True
        for b in [1,2]:
            h=by['H0_static_r'+str(b)][k];m=by['experience_review_r'+str(b)][k]
            good=good and h.get('predicted')!=gold[k] and m.get('initial_predicted') in list('ABCDEF') and m['initial_predicted']!=gold[k] and m.get('review_predicted')==gold[k] and m.get('predicted')==gold[k]
        if good:repaired.append(k)
    passed=complete and len(repaired)>=2 and all(p['H0_to_experience']['gain']-p['H0_to_experience']['loss']>=2 and p['H0_to_experience']['loss']<=1 and p['generic_to_experience']['gain']-p['generic_to_experience']['loss']>=1 for p in pairs)
    return {'pass':bool(passed),'repeat_observed_attributable_repair_count':len(repaired),'repeat_observed_attributable_repair_ids':repaired,'exploratory_error_enriched_development':True}
