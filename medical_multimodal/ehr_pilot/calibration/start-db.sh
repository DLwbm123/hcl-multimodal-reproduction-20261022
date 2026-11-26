#!/bin/bash
set -eu
umask 077
cd "$(dirname "$0")"
root="$PWD"
test -s "$PREPARED/config.yaml"
ulimit -f 12582912
mkdir -p data
cp --reflink=auto "$PREPARED/data/test_db.mv.db" data/
sed "s|jdbc:h2:[^;']*|jdbc:h2:$root/data/test_db|" "$PREPARED/config.yaml" > config.yaml
export SPRING_CONFIG_LOCATION="file:$root/config.yaml"
cd "$PREPARED/app"
exec "$PREPARED/usr/lib/jvm/java-17-openjdk-amd64/bin/java" -Xmx3g -XX:ActiveProcessorCount=4 --class-path main.war '-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/' org.springframework.boot.loader.PropertiesLauncher --server.address=127.0.0.1 --server.port=18476
