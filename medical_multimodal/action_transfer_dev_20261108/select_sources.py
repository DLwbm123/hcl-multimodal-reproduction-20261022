"""Reproduce the metadata-only development selection; write results privately.

Exclude the entire case before inspecting its contents. The caller supplies
case identifiers from the label-free historical member/input/panel manifests.
Image privacy review is a separate mandatory step before model execution.
"""
import re


def select_sources(source_metadata, excluded_case_ids):
    excluded = set(map(str, excluded_case_ids))
    negative, positive = [], []
    for cid in sorted(source_metadata, key=int):
        if cid in excluded:
            continue
        source = source_metadata[cid]
        for figure in source.get("figures", []):
            for view in figure["subfigures"]:
                caption = view["caption"]
                if not re.search(r"(?i)(chest|cxr|frontal|pa |antero)", caption):
                    continue
                if not re.search(r"(?i)(radiograph|x.?ray|cxr|plain.*film)", caption):
                    continue
                normal = bool(re.search(r"(?i)normal (?:chest |pa |frontal )?(?:radiograph|x.?ray|cxr)|(?:chest (?:x.?ray|radiograph)).{0,55}(?:was|is|appears) normal", caption))
                normal = normal and not re.search(r"(?i)abnormal|except|otherwise|normal (?:heart|cardiac)|cardiac.*normal", caption)
                abnormal = bool(re.search(r"(?i)air.?space (?:opacit|shadow|consolida)|(?:patchy|bilateral|lobar|alveolar) .{0,25}consolidat|consolidat.{0,50}(?:lung|lobe)", caption))
                pool = negative if normal and not abnormal else positive if abnormal and not normal else None
                if pool is not None and not any(x["source_group"] == cid for x in pool):
                    pool.append({"source_group": cid, "figure": view["number"],
                                 "url": view["url"], "caption": caption,
                                 "source_url": source["link"], "reference": int(pool is positive)})
    if min(len(negative), len(positive)) < 3:
        raise ValueError("Insufficient eligible source groups; no silent relaxation")
    selected = [pool[i] for i in range(3) for pool in [negative, positive]]
    if len({x["source_group"] for x in selected}) != 6:
        raise ValueError("Cross-stratum source overlap requires a new protocol decision")
    return selected
