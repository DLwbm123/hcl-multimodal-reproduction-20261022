# One controlled tool-consultation diagnostic

The preceding native MedRAX/local-Qwen run completed 12 questions with four
correct answers, but used a real tool in only one episode and failed its frozen
qualification gate. All four tools themselves executed successfully.

This independent development diagnostic tests one hypothesis with two paired
arms on 12 new cases: retain the native request, or append one instruction to
consult a chest X-ray tool before answering and assess its relevance. The native
graph, system prompt, tool schemas, assets, decoding and episode budgets are
identical. No memory, routing state, backbone weights or HCL component updates.
Arm order alternates; there is no conversational carryover.

`protocol.json` freezes both the fresh panel and the sole candidate instruction.
All old and new development cases are excluded from future validation and test
sets. Gold answers/explanations are used only after all 24 planned episodes.
Failures and unrun episodes remain in the denominator. No candidate prompt search
or repeated changes on this panel are permitted.

Prefer the native arm if it passes the original gate. Otherwise the consultation
arm must have at least three correct answers, four valid finals, six episodes
with successful real tools, no episode runtime errors, actual visual payloads,
and at least as many correct answers as its paired native arm. If neither passes,
close this prompt-adaptation branch. Passing is a feasibility decision for a
common base agent, not a claim about HCL or medical accuracy.

The experiment has its own one-hour limit, additionally bounded by the existing
route deadline. It reuses existing assets/environments read-only. Aggregate new
storage for the route remains below 30 GB, with a 28 GB early stop including
reserve. Its root and neutral worker receipts must be registered in the existing
route storage policy before launch. The original failed native run stays sealed.

`run_paired.py` uses the sibling `medrax_v1` runtime; deployment copies its pinned
source alongside the driver. `control.py` supplies independent resource guards.
Runtime configuration, data, images, weights and full traces remain private.
