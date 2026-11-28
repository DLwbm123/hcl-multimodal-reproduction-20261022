#!/usr/bin/env bash
# Start a private, disposable copy of the prepared snapshot on loopback.
set -euo pipefail
umask 077
test -z "$(ss -H -ltn 'sport = :18489')"
ulimit -f 6200000
mkdir -p data
if ! test -s data/test_db.mv.db; then
    cp --reflink=auto i0/data/test_db.mv.db data/seed.tmp
    mv data/seed.tmp data/test_db.mv.db
fi
sed "s|jdbc:h2:[^;']*|jdbc:h2:$PWD/data/test_db|" i0/config.yaml |
    sed "s/;DB_CLOSE_DELAY=-1/;AUTO_COMPACT_FILL_RATE=0;MAX_COMPACT_TIME=0;DB_CLOSE_DELAY=-1/" > config.yaml
export SPRING_CONFIG_LOCATION="file:$PWD/config.yaml"
exec ./j/bin/java -Xmx3g -XX:ActiveProcessorCount=4 --class-path main.war "-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/" org.springframework.boot.loader.PropertiesLauncher --server.address=127.0.0.1 --server.port=18489
