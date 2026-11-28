"""Reuse the frozen panel for ranking and full-frame resolution diagnostics."""
import json
import os
from pathlib import Path
from statistics import median, mean
import time
from zipfile import ZipFile

from probe import ACTIONS, Reader, load, views

SIDES = (448, 768, 1024)
BATCH = 2


def operating_point(positives, negatives, threshold):
    tp = sum(p > threshold for p in positives)
    tn = sum(p <= threshold for p in negatives)
    sensitivity = tp / len(positives)
    specificity = tn / len(negatives) if negatives else None
    return dict(threshold=threshold, tp=tp, fn=len(positives)-tp, tn=tn if negatives else None,
                fp=len(negatives)-tn if negatives else None, sensitivity=sensitivity,
                specificity=specificity,
                balanced_accuracy=(sensitivity+specificity)/2 if negatives else None)


def classification(positives, negatives):
    assert positives and all(0 <= p <= 1 for p in positives + negatives)
    half = operating_point(positives, negatives, .5)
    if not negatives:
        return dict(n_positive=len(positives), n_negative=0, auc=None, at_half=half, in_panel_best=None)
    values = sorted(set(positives + negatives))
    candidates = [0., .5, 1.] + [(a+b)/2 for a,b in zip(values,values[1:])]
    fitted = max((operating_point(positives, negatives, t) for t in candidates),
                 key=lambda p:(p['balanced_accuracy'], p['specificity'], -abs(p['threshold']-.5)))
    pairs = sum((a > b) + .5*(a == b) for a in positives for b in negatives)
    return dict(n_positive=len(positives), n_negative=len(negatives),
                auc=pairs/(len(positives)*len(negatives)), at_half=half, in_panel_best=fitted)


def scores(rows, action):
    return ([r['scores'][action] for r in rows if r['label'] and action in r['scores']],
            [r['scores'][action] for r in rows if not r['label'] and action in r['scores']])


def token_counts(reader, encoded):
    counts = (encoded.input_ids == reader.model.config.image_token_id).sum(dim=1).tolist()
    merge = reader.processor.image_processor.merge_size
    expected = sum(int(d.prod()) // merge**2 for d in encoded.image_grid_thw)
    assert sum(counts) == expected and min(counts)>0
    return counts


def tokens(values):
    return dict(min=min(values), median=median(values), mean=mean(values), max=max(values))


def paired_pos(labels, reference, candidate):
    return dict(rescued=sum(y and a<=.5<b for y,a,b in zip(labels,reference,candidate,strict=True)),
                harmed=sum(y and b<=.5<a for y,a,b in zip(labels,reference,candidate,strict=True)))


def save(name, value):
    path=Path(name)
    staging=path.with_suffix('.tmp')
    staging.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    staging.replace(path)


def main():
    start=time.monotonic()
    source=Path(os.environ['SOURCE_RUN'])
    assert (source/'exits').read_text().strip()=='0'
    panel=json.loads((source/'private_panel.json').read_text())
    history=[json.loads(line) for line in (source/'private_predictions.jsonl').read_text().splitlines()]
    assert len(panel)==len(history)==96 and len({r['member'] for r in panel})==96
    assert all(all(a[k]==b[k] for k in ('member','label','mask','category'))
               for a,b in zip(panel,history,strict=True))
    labels=[r['label'] for r in panel]
    assert sum(labels)==48
    offline={action:classification(*scores(history,action)) for action in (*ACTIONS,'lesion')}
    assert offline['direct']['at_half']['tp']==11 and offline['lesion']['at_half']['tp']==20
    save('offline.json',offline)
    print('offline ready',flush=True)
    root=Path(os.environ['DATA_ROOT'])
    with ZipFile(root/'hyper-kvasir-labeled-images.zip') as archive:
        images=[load(archive,r['member'],'RGB') for r in panel]
    reader=Reader()
    from transformers import AutoProcessor

    any_448=[r['scores']['direct'] for r in history]
    lesion=[r['scores'].get('lesion') for r in history]
    report={}
    for side in SIDES:
        if side!=448:
            reader.processor=AutoProcessor.from_pretrained(os.environ['MODEL'],local_files_only=True,
                                      min_pixels=56*56,max_pixels=side*side)
            reader.processor.tokenizer.padding_side='left'
        assert reader.processor.image_processor.max_pixels==side**2
        measured=[]
        predictions=[]
        for start_at in range(0,len(images),BATCH):
            batch=[('direct',[im]) for im in images[start_at:start_at+BATCH]]
            encoded=reader.encode(batch)
            measured.extend(token_counts(reader,encoded))
            predictions.extend(reader.probabilities(encoded))
        pos=[p for p,y in zip(predictions,labels,strict=True) if y]
        neg=[p for p,y in zip(predictions,labels,strict=True) if not y]
        report[str(side)] = dict(classification=classification(pos,neg),
                tokens_all=tokens(measured), tokens_positive=tokens([t for t,y in zip(measured,labels,strict=True) if y]),
                paired_positive_vs_rerun_448=paired_pos(labels,predictions if side==448 else baseline,predictions),
                paired_positive_vs_lesion=paired_pos([True]*len(pos), [p for p in lesion if p is not None], pos))
        if side==448:
            baseline=predictions
            audit=dict(max_abs_difference=max(abs(a-b) for a,b in zip(any_448,baseline,strict=True)),
              decisions_changed=sum((a>.5)!=(b>.5) for a,b in zip(any_448,baseline,strict=True)))
            measured_448=measured
            with ZipFile(root/'hyper-kvasir-segmented-images.zip') as masks:
                total_lesion=[]
                for row,image in zip(panel,images,strict=True):
                    if row['label']:
                        mask=load(masks,row['mask'],'L')
                        visual,_,_=views(image,mask)
                        total_lesion.extend(token_counts(reader,reader.encode([('lesion',visual['lesion'])])))
            lesion_cost=tokens(total_lesion)
        assert all(a<=b for a,b in zip(measured_448,measured,strict=True))
        print('finished side',side,'examples',len(predictions),flush=True)
    save('diagnostics.json',dict(offline=offline,full=report,lesion_tokens_total=lesion_cost,
        baseline_audit=audit,seconds=round(time.monotonic()-start,3),
        max_reserved_mib=reader.torch.cuda.max_memory_reserved()/2**20,
        native_frames_larger_than_448_area=sum(im.width*im.height>448**2 for im in images)))


if __name__=='__main__':
    main()
