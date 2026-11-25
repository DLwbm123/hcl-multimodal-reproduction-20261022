"""Counterbalance the first tool menu; compare third-observation outcomes on fit cases."""
import json
import os
from collections import Counter
from pathlib import Path

from acquire import load_vlm
from prep import select
from visual_ops import VisualTools, OPEN, CROP, COMPARE, multi_episode

root = Path(os.environ["RUN_ROOT"])
original = Path(os.environ["SOURCE_RUN"])
seed = json.loads((original/"design.json").read_text())["seed_instruction"]
menu_text = "A: open_view\nB: inspect_region"


def main():
    choose = load_vlm()
    records, _, _ = select(Path(os.environ["RAW_ROOT"]))
    rows = []
    for stage in ("A","B"):
        directory = original / (stage+"_fit")
        case_ids = {p.name for p in (directory/"images").iterdir()}
        cases = [r for r in records if r[0] in case_ids]
        if len(cases)!=16:
            raise ValueError("Incorrect frozen fit panel")
        for case_id, _, history, options, gold, views in cases:
            cat = [dict(view=f"view{i:02}",modality=v["modality"],type=v["type"]) for i,v in enumerate(views,1)]
            case = dict(case=case_id,history=history,options=options,views=cat)
            row = {"stage":stage,"menu":{},"diagnoses":{}}
            class Finished(Exception):
                pass
            def capture(context,images,instruction,labels):
                if labels!=["A","B"] or not instruction.endswith(menu_text):
                    raise ValueError("Upstream menu changed")
                for swapped in (False,True):
                    mapping = {"A":CROP if swapped else OPEN,"B":OPEN if swapped else CROP}
                    for reversed_order in (False,True):
                        order = ["B","A"] if reversed_order else ["A","B"]
                        menu = "\n".join(k+": "+mapping[k] for k in order)
                        prompt = instruction[:-len(menu_text)]+menu
                        decision = choose(context,images,"Procedure:\n"+seed+"\nCurrent action:\n"+prompt,labels)
                        row["menu"][f"swap{int(swapped)}_reverse{int(reversed_order)}"] = mapping[decision]
                shared["context"] = context
                raise Finished
            shared = {}
            try:
                multi_episode(capture,directory,case)
            except Finished:
                pass
            for name,args in [(OPEN,("view03",)),(CROP,("view02","center")),(COMPARE,("view01","view02","center"))]:
                t=VisualTools(directory,case)
                t.invoke(OPEN,("view01",))
                t.invoke(OPEN,("view02",))
                t.invoke(name,args)
                inst="Use the shown observations and history to choose the most likely diagnosis. Return a single option letter."
                pred=choose(shared["context"],t.shown,"Procedure:\n"+seed+"\nCurrent action:\n"+inst,list(options))
                row["diagnoses"][name]=pred
            row.update(case=case_id,gold=gold)
            rows.append(row)
    out = {"n_fit":len(rows),"first_menu":{},"third_observation":{}}
    for mode in rows[0]["menu"]:
        out["first_menu"][mode] = dict(Counter(r["menu"][mode] for r in rows))
    for name in (OPEN,CROP,COMPARE):
        gain=sum(r["diagnoses"][name]==r["gold"]!=r["diagnoses"][OPEN] for r in rows)
        loss=sum(r["diagnoses"][OPEN]==r["gold"]!=r["diagnoses"][name] for r in rows)
        out["third_observation"][name] = dict(correct=sum(r["diagnoses"][name]==r["gold"] for r in rows),gain=gain,loss=loss)
    out["oracle_correct"] = sum(r["gold"] in r["diagnoses"].values() for r in rows)
    (root/"private_rows.json").write_text(json.dumps(rows))
    (root/"result.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":
    main()
