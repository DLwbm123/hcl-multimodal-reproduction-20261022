"""Prepare a paired three-view pilot from the pinned official MedThinkVQA release."""
import json
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


REVISION = "3a0b076f5116c6237ca94bb71fc14066cddf7cdc"
SEED = 20261125
SPLITS = {"development": 32, "heldout": 64}
SLOT_PATH = re.compile(r"image_(\d{2})_path")
IMAGE_PATH = re.compile(r"images/(case\d+)/([A-Za-z0-9_-]+)\.jpg")


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def read_rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def views(row):
    result = []
    for key in sorted(row):
        slot = SLOT_PATH.fullmatch(key)
        if not slot or not row[key]:
            continue
        path = IMAGE_PATH.fullmatch(row[key])
        if path is None:
            raise ValueError(f"Unexpected image path {row[key]!r}")
        prefix = "image_" + slot[1]
        result.append({
            "case": path[1], "image": path[2], "path": row[key],
            "modality": row[prefix + "_modality"], "type": row[prefix + "_sub_modality"],
        })
    if not result or len({v["case"] for v in result}) != 1:
        raise ValueError("Missing or ambiguous case identifier")
    return result


def keys(history, case_views):
    return {("history", normalize(history)), ("case", case_views[0]["case"])} | {
        ("image", v["image"]) for v in case_views
    }


def select(root):
    reserved = set()
    test_count = 0
    all_paths = set()
    for row in read_rows(root / "test.jsonl"):
        case_views = views(row)
        reserved.update(keys(row["CLINICAL_HISTORY"], case_views))
        all_paths.update(v["path"] for v in case_views)
        test_count += 1

    candidates = []
    counts = Counter()
    train_count = 0
    for row in read_rows(root / "train.jsonl"):
        train_count += 1
        case_views = views(row)
        all_paths.update(v["path"] for v in case_views)
        history = row["CLINICAL_HISTORY"].strip()
        identity = keys(history, case_views)
        # Exclude presentation figures and screenshots that can contain printed diagnoses.
        usable = [v for v in case_views if v["modality"] and v["type"] and
                  v["modality"] != "Non-modality / Workflow / Post-processing" and
                  v["type"] not in {"Annotated figure / diagram", "PACS / Teleradiology screenshot"}]
        options = row["options"]
        answer = row["correct_answer"]
        history_words = " " + normalize(history) + " "
        if reserved & identity:
            counts["matches_official_test_identifier_image_or_history"] += 1
        elif len(usable) < 4 or len({v["image"] for v in usable}) != len(usable):
            counts["insufficient_distinct_usable_views"] += 1
        elif len({(v["modality"], v["type"]) for v in usable[1:]}) < 2:
            counts["no_described_choice_after_first_view"] += 1
        elif not history or answer not in options or any(not normalize(s) for s in options.values()):
            counts["incomplete_question"] += 1
        elif any(" " + normalize(s) + " " in history_words for s in options.values()):
            counts["option_text_in_history"] += 1
        else:
            candidates.append((case_views[0]["case"], identity, history, options, answer, usable))

    candidates.sort(key=lambda item: item[0])
    random.Random(SEED).shuffle(candidates)
    seen = set()
    unique = []
    for record in candidates:
        if record[1] & seen:
            counts["matches_other_eligible_case_image_or_history"] += 1
            continue
        unique.append(record)
        seen.update(record[1])
    if (train_count, test_count) != (7347, 720) or len(unique) < sum(SPLITS.values()):
        raise ValueError("Unexpected release or insufficient eligible cases")
    summary = {"revision": REVISION, "seed": SEED, "train": train_count, "test": test_count,
               "eligible_after_deduplication": len(unique), "excluded": dict(counts),
               "official_referenced_images": len(all_paths), "panels": {}}
    return unique, all_paths, summary


def write_row(f, obj):
    print(json.dumps(obj, ensure_ascii=False), file=f)


def prepare(root):
    records, expected, summary = select(root)
    output = root / "pilot"
    if output.exists():
        raise FileExistsError(f"Pilot already exists at {output}")
    offset = 0
    with tempfile.TemporaryDirectory(dir=root) as temp, ZipFile(root / "images.zip") as archive:
        missing = expected.difference(archive.namelist())
        if missing:
            raise ValueError(f"{len(missing)} official image paths missing, e.g. {min(missing)}")
        dest = Path(temp) / "pilot"
        dest.mkdir()
        for split, count in SPLITS.items():
            panel = dest / split
            panel.mkdir()
            media = panel / "images"
            media.mkdir()
            image_count = 0
            with (panel / "inputs.jsonl").open("x") as prompts, (panel / "answers.jsonl").open("x") as answers:
                for case, _, history, options, answer, usable in records[offset:offset + count]:
                    catalog = []
                    (media / case).mkdir()
                    for number, view in enumerate(usable, 1):
                        name = f"view{number:02d}"
                        with archive.open(view["path"]) as src:
                            payload = src.read()
                        # Some upstream files carry a .jpg suffix around PNG data.
                        with Image.open(BytesIO(payload)) as img:
                            img.load()
                            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(img.format)
                            if not ext:
                                raise ValueError(f"Unsupported image format {img.format}")
                        (media / case / (name + ext)).write_bytes(payload)
                        catalog.append({"view": name, "modality": view["modality"], "type": view["type"]})
                    write_row(prompts, {"case": case, "history": history, "options": options, "views": catalog})
                    write_row(answers, {"case": case, "answer": answer})
                    image_count += len(catalog)
            summary["panels"][split] = {"cases": count, "decoded_images": image_count}
            offset += count
        (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        dest.rename(output)
    return summary


if __name__ == "__main__":
    os.umask(0o077)
    print(json.dumps(prepare(Path(os.environ["DATA_ROOT"])), indent=2))
