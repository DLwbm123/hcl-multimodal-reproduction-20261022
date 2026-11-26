set -eu
umask 077
: "${PANEL_ROOT:?}" "${RUN_GPU:?}" "${PART:?}" "${PILOT_ROOT:?}" "${LIB_RUN:?}" "${FEATURE_RUN:?}" "${MODEL_ROOT:?}" "${REFEREE1:?}" "${REFEREE2:?}" "${BOUNDED_RUN:?}" "${OUTCOME_FIT_CACHE:?}" "${HCL_SOURCE:?}" "${CTA_ROOT:?}" "${DATA_ROOT:?}"
export RUN_ROOT="$PANEL_ROOT/part$PART" STUDY=clinical
export PYTHONPATH="$RUN_ROOT:$HCL_SOURCE"
mkdir "$RUN_ROOT"
cd "$RUN_ROOT"
cp "$LIB_RUN"/*.py .
cp "$FEATURE_RUN/run.py" feature_policy.py
cp "$PANEL_ROOT"/{run.py,check.py,design.json} .
python3 - <<'CODE'
from pathlib import Path
import json,time,os
p=Path('design.json')
d=json.loads(p.read_text())
d['budget']['max_seconds']=int((Path(os.environ['PANEL_ROOT'])/'start_epoch').read_text())+7200-int(time.time())-15
assert d['budget']['max_seconds']>120
p.write_text(json.dumps(d))
CODE
python3 -B start.py
