set -eu
umask 077
: "${RUN_ROOT:?}" "${RUN_GPU:?}" "${FEATURE_RUN:?}" "${QUALITY_RUN:?}" "${PILOT_ROOT:?}" "${MODEL_ROOT:?}" "${CTA_ROOT:?}" "${DATA_ROOT:?}" "${HCL_SOURCE:?}"
export STUDY=clinical PYTHONPATH="$RUN_ROOT:$HCL_SOURCE"
test "$(findmnt -n -T "$(dirname "$RUN_ROOT")" -o TARGET)" = /data
mkdir "$RUN_ROOT"
cd "$RUN_ROOT"
cp "$FEATURE_RUN"/*.py .
tar -xf -
python3 -B freeze.py
python3 - <<'CODE'
from pathlib import Path
p=Path('acquire.py')
s=p.read_text()
for anchor, event in [('scores = model(**batch', "('choice', int(batch.input_ids.shape[1]), 0)"),
                      ('return processor.tokenizer.decode(tokens', "('generate', int(inputs.input_ids.shape[1]), int(tokens.shape[1] - inputs.input_ids.shape[1]))")]:
    assert s.count(anchor)==1
    line=next(x for x in s.splitlines() if anchor in x)
    indent=line[:len(line)-len(line.lstrip())]
    s=s.replace(line,indent+"with (ROOT/'tokens.jsonl').open('a') as f:\n"+indent+"    print(json.dumps("+event+"), file=f)\n"+line)
p.write_text(s)
CODE
python3 -B start.py
