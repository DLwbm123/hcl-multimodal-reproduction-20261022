import json
from pathlib import Path
from zipfile import ZipFile

from prep import select


def build(raw, root, design):
    pool, _, _ = select(raw)
    old = set()
    for name in ("development", "heldout"):
        for line in (raw / "pilot" / name / "inputs.jsonl").read_text().splitlines():
            old.add(json.loads(line)["case"])
    if {item[0] for item in pool[:96]} != old:
        raise ValueError("Previous pilot does not match")
    result = {}
    with ZipFile(raw / "images.zip") as archive:
        for stage in ("A","B"):
            rows = [r for r in pool[96:] if len(r[-1]) <= 27 and
                    any(v["modality"] == "CT" for v in r[-1]) == (stage == "A")]
            start = 0
            for role, n in design["roles"].items():
                directory = root / (stage + "_" + role)
                directory.mkdir()
                inputs, answers = [], {}
                for case, _, history, options, answer, views in rows[start:start+n]:
                    if case in old:
                        raise ValueError("Case split overlap")
                    old.add(case)
                    dest = directory / "images" / case
                    dest.mkdir(parents=True)
                    catalog = []
                    for i,v in enumerate(views,1):
                        key = f"view{i:02d}"
                        (dest / (key + ".img")).write_bytes(archive.read(v["path"]))
                        catalog.append({"view":key,"type":v["type"],"modality":v["modality"]})
                    inputs.append(dict(case=case,history=history,options=options,views=catalog))
                    answers[case] = answer
                if len(inputs) != n:
                    raise ValueError("Incomplete split")
                (directory / "gold.json").write_text(json.dumps(answers))
                result[stage,role] = (directory,inputs)
                start += n
    return result
