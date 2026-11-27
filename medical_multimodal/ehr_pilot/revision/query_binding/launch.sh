#!/bin/bash
set -eu
umask 077
. ./env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Bound the isolated database file while retaining room for H2 metadata.
ulimit -f 5700000
if ss -ltn | grep -qw 18478; then exit 3; fi
mkdir -p data
if ! test -f data/test_db.mv.db; then
    cp --reflink=auto archive/data/test_db.mv.db data/seed.tmp
    mv data/seed.tmp data/test_db.mv.db
fi
sed "s|jdbc:h2:[^;']*|jdbc:h2:$PWD/data/test_db|" archive/config.yaml | sed "s/;DB_CLOSE_DELAY=-1/;AUTO_COMPACT_FILL_RATE=0;MAX_COMPACT_TIME=0;DB_CLOSE_DELAY=-1/" > config.yaml
export SPRING_CONFIG_LOCATION="file:$PWD/config.yaml"
./j/bin/java -Xmx3g -XX:ActiveProcessorCount=4 --class-path main.war '-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/' org.springframework.boot.loader.PropertiesLauncher --server.address=127.0.0.1 --server.port=18478 > db.log 2>&1 &
db=$!
trap 'kill -TERM "$db" 2>/dev/null || true' EXIT
for i in $(seq 1 100); do
    if curl -fsS --max-time 5 -o /dev/null http://127.0.0.1:18478/fhir/metadata 2>/dev/null; then break; fi
    kill -0 "$db"
    sleep 2
done
curl -fsS --max-time 10 -o /dev/null http://127.0.0.1:18478/fhir/metadata
ulimit -f 32768
devices=("$@")
workers=()
for i in 0 1 2 3; do
    PART="$i" CUDA_VISIBLE_DEVICES="${devices[i]}" python -u t.py eval > "part$i.log" 2>&1 &
    workers+=("$!")
done
status=0
for w in "${workers[@]}"; do wait "$w" || status=1; done
test "$status" = 0
python t.py report
test -s result.json
