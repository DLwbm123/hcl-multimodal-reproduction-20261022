set -eu
umask 077
: "${RUN_ROOT:?new private output directory}" "${RUN_GPU:?GPU index}"
: "${QUALITY_RUN:?completed visual-observation run}" "${VETO_RUN:?earlier stop-policy run}"
: "${BOUNDED_RUN:?frozen trust and stopping rules}" "${OUTCOME_FIT_CACHE:?original fitting memory}"
: "${PAIRED_RUN:?completed repaired baseline}" "${PILOT_ROOT:?pilot environment}"
: "${MODEL_ROOT:?shared frozen checkpoint}" "${CTA_ROOT:?MedCTA preparation}" "${DATA_ROOT:?prepared images and tasks}"
: "${HCL_SOURCE:?directory containing the hcl Python package}"
export RUN_ROOT RUN_GPU QUALITY_RUN VETO_RUN BOUNDED_RUN OUTCOME_FIT_CACHE PAIRED_RUN PILOT_ROOT MODEL_ROOT CTA_ROOT DATA_ROOT
export STUDY=clinical PYTHONPATH="$RUN_ROOT:$HCL_SOURCE"
mkdir "$RUN_ROOT"
cd "$RUN_ROOT"
cp "$QUALITY_RUN"/*.py .
tar -xf -
# Both visual tools are served by observation.py. The original RPC worker supplies OCR and arithmetic.
sed -i "s/for name in ('OCR','ImageDescription','RegionAttributeDescription'):/for name in ('OCR',):/" tool_server.py
python3 -B start.py
