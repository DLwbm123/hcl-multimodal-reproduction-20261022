"""Conditional procedural experience; generation acceptance is not expert validation."""
import json
import re

ARMS=['H0_static','generic_procedure','H_A']
FIELDS=['applicability','procedure','do_not_apply','supervision_limit']
STOP=set('the a an of to in on and or is are was were be this that these those which what with for from as by at it its image images imaging chest xray ray radiograph radiographs provided following based finding findings patient most likely correct answer option'.split())
GENERIC={
 'applicability':'Apply these general checks when comparing the supplied answer choices.',
 'procedure':'First identify the visible finding and its location. Distinguish observation from etiology and confirmed diagnosis. Compare the exact differences between competing options, including side, anatomical zone, measurement, extent and certainty. Infer exact dimensions only when an adequate scale or measurement is supplied. Use current images and stated clinical information; consider an option that makes an unsupported specific assertion less well supported. Treat tool outputs as fallible secondary evidence and reconcile them with the original images.',
 'do_not_apply':'Do not import facts from another patient, an absent view, an unstated follow-up or external final diagnosis. Do not replace image evidence with a general rule or invent findings to support a preferred option.',
 'supervision_limit':'These are fixed general procedures, constructed before the development allocation, without using development questions, answers or outcomes.'}


def parse_note(text):
    clean=text.strip()
    if clean.startswith('```'): clean=re.sub(r'^```(?:json)?\s*|\s*```$','',clean)
    value=json.loads(clean)
    assert set(value)==set(FIELDS+['admit','reason','cue_terms'])
    assert type(value['admit']) is bool and isinstance(value['reason'],str) and 0<len(value['reason'])<=600
    assert isinstance(value['cue_terms'],list) and 2<=len(value['cue_terms'])<=12
    assert all(isinstance(s,str) and 0<len(s)<=60 for s in value['cue_terms'])
    assert all(isinstance(value[k],str) and value[k].strip() for k in FIELDS)
    assert sum(len(value[k]) for k in FIELDS)<=1600
    return value


def tokens(text):return set(re.findall('[a-z0-9]+',text.lower()))-STOP


def render(snapshot,question,train_ids):
    assert not snapshot['raw'] and not snapshot['skills'] and not snapshot['anchor']
    records=snapshot['abstract']
    if not records:return '',[]
    if len(records)==1 and records[0]['memory_id']=='fixed_generic':chosen=records
    else:
        assert all(r['internal_source']['task_id'] in train_ids for r in records)
        assert all(set(r.get('source_ids',[r['internal_source']['task_id']])) <= train_ids for r in records)
        q=tokens(question);rank=[]
        for r in records:
            cues=tokens(' '.join(r['cue_terms']));overlap=q&cues
            if len(overlap)>=2:rank.append((-(len(overlap)/len(q|cues)),r['memory_id'],r))
        chosen=[v[2] for v in sorted(rank)[:2]]
    if not chosen:return '',[]
    views=[{k:r['note'][k] for k in FIELDS} for r in chosen]
    text='\nHistorical procedural guidance. These are fallible candidate procedures, not findings or answers for this case. Use only when the applicability and current image evidence agree. Training supervision was not clinically adjudicated; do not invent facts to justify it.\n'+json.dumps(views,ensure_ascii=False)
    return text,[r['memory_id'] for r in chosen]


def repeated_gate(pairs,reference,generic,learned,ids,gold,complete):
    # Fixed exploration gate, not a test of population-level clinical benefit.
    acquired=[k for k in ids if all(reference[j][k].get('predicted')!=gold[k] and learned[j][k].get('predicted')==gold[k] for j in range(2))]
    passes=complete and len(acquired)>=2 and all(p['H0_to_HA']['gain']-p['H0_to_HA']['loss']>=2 and p['H0_to_HA']['loss']<=1 and p['generic_to_HA']['gain']-p['generic_to_HA']['loss']>=1 for p in pairs)
    return {'pass':bool(passes),'repeat_observed_acquired_count':len(acquired),'repeat_observed_acquired_ids':acquired,'gate_is_exploratory_not_confirmation':True}
