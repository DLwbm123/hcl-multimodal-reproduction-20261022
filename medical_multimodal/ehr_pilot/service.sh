#!/bin/bash
set -eu
cd "$(dirname "$0")"
umask 077
# One isolated copy. The source OCI directory is read only.
cd runtime
src="$SOURCE/fhir/blobs/sha256"
ulimit -f 12582912
for prefix in f986 cf24 8c20 9bfc; do
    archive=$(compgen -G "$src/$prefix*")
    tar xzf "$archive" --no-same-owner
done
python3 - <<'FIX'
import os
from pathlib import Path
root = Path.cwd()
for link in (root / 'usr/lib/jvm').rglob('*'):
    if link.is_symlink():
        target = os.readlink(link)
        if target.startswith('/') and (root / target[1:]).exists():
            link.unlink()
            link.symlink_to(root / target[1:])
FIX
java=$(find usr/lib/jvm -path '*/bin/java' | head -1)
root=$PWD
db="$root/data/test_db"
test -s "$db.mv.db"
sed "s|jdbc:h2:/data/test_db|jdbc:h2:$db|" configs/application.yaml > config.yaml
export SPRING_CONFIG_LOCATION="file:$root/config.yaml"
cd app
exec "$root/$java" -Xmx3g -XX:ActiveProcessorCount=4 --class-path main.war '-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/' org.springframework.boot.loader.PropertiesLauncher --server.address=127.0.0.1 --server.port=18476
