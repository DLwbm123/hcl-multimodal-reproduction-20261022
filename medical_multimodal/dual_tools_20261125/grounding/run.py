import json
from grounding import boxes, iou, locate, reference_box, summarize
from med import main
from shared import save


class FitAudit:
    def __init__(self):
        self.rows=[]

    def __call__(self, key, image, dialogs, notes, response, score):
        args=next(t['function']['arguments'] for m in dialogs for t in m.get('tool_calls',[])
                  if t['function']['name']=='RegionAttributeDescription')
        truth,clipped=reference_box(args,image.size)
        predicted=json.loads(response['arg'])['box'] if response['ok'] else None
        menu=boxes(image.size)
        self.rows.append(dict(case=key,usable=response['ok'] and bool(response['text'].strip()),
                              clipped=clipped,iou=iou(predicted,truth) if truth and predicted else None,
                              center=iou(menu[4],truth) if truth else None,
                              best=max(iou(b,truth) for b in menu) if truth else None,
                              before=score(notes),after=score([*notes,response])))

    def finish(self):
        save('private_fit_region',self.rows)
        save('fit_audit',summarize(self.rows))


def search_panel(cases, refs, used):
    seen={p for k in used for p in cases[k]['images']}
    ids=sorted(k for k in set(refs)-set(used) if seen.isdisjoint(cases[k]['images']))
    return [('search_stress',ids)] if ids else []


main(locator=locate,fit_observer=FitAudit(),extra_panels=search_panel)
