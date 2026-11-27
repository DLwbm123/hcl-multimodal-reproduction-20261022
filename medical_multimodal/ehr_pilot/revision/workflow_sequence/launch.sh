#!/usr/bin/env bash
set -euo pipefail
umask 077
. ./env
export PYTHONDONTWRITEBYTECODE=1
mkdir -p cache/tmp data
export TMPDIR="$PWD/cache/tmp" XDG_CACHE_HOME="$PWD/cache" HF_HOME="$PWD/cache/hf"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torch"
own=$$
pids=()
cleanup() {
    if (("${#pids[@]}")); then kill -TERM "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
(
    while kill -0 "$own" 2>/dev/null; do
        if (( $(du -sB1 . | cut -f1) >= 7500000000 )); then
            echo "storage guard reached"
            kill -TERM "$own"
            break
        fi
        sleep 5
    done
) &
pids+=("$!")
if [[ ! -s data/test_db.mv.db ]]; then
    cp --sparse=always db_source/data/test_db.mv.db data/db.part
    mv data/db.part data/test_db.mv.db
fi
sed -E "s#jdbc:h2:[^;']*/test_db#jdbc:h2:./data/test_db#g" db_source/config.yaml > config.yaml
export SPRING_CONFIG_LOCATION="file:./config.yaml"
./j/bin/java -Xmx3g -XX:ActiveProcessorCount=4 --class-path main.war \
    '-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/' \
    org.springframework.boot.loader.PropertiesLauncher --server.address=127.0.0.1 \
    --server.port=18476 > service.log 2>&1 &
pids+=("$!")
for tick in $(seq 1 150); do
    if curl -fsS --max-time 3 http://127.0.0.1:18476/fhir/metadata > /dev/null 2>&1; then break; fi
    kill -0 "${pids[1]}"
    sleep 2
done
curl -fsS --max-time 5 http://127.0.0.1:18476/fhir/metadata >/dev/null
for part in 0 1; do
    CUDA_VISIBLE_DEVICES=$((6 + part)) PART=$part python -u t.py infer > "$part.log" 2>&1 &
    pids+=("$!")
done
wait "${pids[2]}"
wait "${pids[3]}"
CUDA_VISIBLE_DEVICES="" python -u t.py report > report.log 2>&1 &
pids+=("$!")
wait "${pids[4]}"
