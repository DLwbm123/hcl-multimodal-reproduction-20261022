"""Convert only the two sensor columns of the pinned Glucose-ML HUPA-UCM archive."""
import csv
import datetime as dt
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def convert(archive, output):
    visits = []
    with zipfile.ZipFile(archive) as source:
        names = sorted(n for n in source.namelist() if n.startswith('HUPA-UCM-extracted-glucose-files/') and n.endswith('.csv'))
        assert len(names) == 25
        for name in names:
            unique = defaultdict(set)
            with source.open(name) as raw:
                table = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
                assert {'timestamp','glucose_value_mg_dl'} <= set(table.fieldnames)
                for row in table:
                    if not row['glucose_value_mg_dl'].strip():
                        continue
                    glucose = float(row['glucose_value_mg_dl'])
                    if math.isfinite(glucose):
                        unique[dt.datetime.fromisoformat(row['timestamp'])].add(glucose)
            rows = []
            for stamp, values in sorted(unique.items()):
                if len(values) == 1:
                    group = 58 if 6 <= stamp.hour < 10 else 62 if 18 <= stamp.hour < 22 else 48
                    rows.append([stamp.date().toordinal(),group,next(iter(values))])
            if rows:
                visits.append(dict(person=str(len(visits)),rows=rows))
    Path(output).write_text(json.dumps(visits,separators=(',',':')))
    days = sum(len({d for d,_,_ in v['rows']}) for v in visits)
    print('participants',len(visits),'days',days,'questions',3*days,flush=True)


if __name__=='__main__':
    convert(*sys.argv[1:])
