"""Complete alternatives for the bounded SkillOpt compatibility check."""
TRUST_EDITS={
    'A':'Accept a change only when the returned text states a concrete finding or measurement relevant to the question. Discard vague restatements, unsupported diagnoses, and claims outside the observed region.',
    'B':'Treat a change from a local crop as provisional when neighboring fit returns of that tool spoiled more answers than they corrected. Require corroborating accepted evidence before it overrides a supported whole-image finding.',
    'C':'Check the returned text against the precise quantity, location, or relationship asked about. Accept OCR or calculation only when the characters or operands are identifiable. Retain the prior answer if the result concerns something else.',
    'D':'Compare the two drafts claim by claim. Accept the revision only when it removes a specific unsupported claim or adds an observation answering the question, and leaves previously supported facts intact.'}
STOP_EDITS={
    'A':'Stop once the accepted observation directly supports the requested answer. If the latest return was rejected and the specific target is unresolved, continue with a different source of evidence.',
    'B':'Continue when the draft still lacks a specific finding and similar fit prefixes included a later correction. Stop if the evidence already addresses the target or another observation would only repeat it.',
    'C':'Normally follow the repaired router. Override its stop only for an unresolved named measurement, text element, or anatomical region that a remaining tool can examine. Override a continuation when the requested fact is already supported.',
    'D':'Stop when the accepted draft connects a directly observed finding to the question. Continue only if you can identify a missing fact and an available tool capable of checking that fact.'}
