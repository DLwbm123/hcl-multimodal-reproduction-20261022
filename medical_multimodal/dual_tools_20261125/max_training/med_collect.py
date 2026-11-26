"""Collect previously unused training feedback with the existing four-tool policy."""
import os
from clinical import actions
from feature_policy import fit_features
from grounding import locate
from med import main
from observation import visual_tools
from shared import save


def collect(cases,refs,parts,context,original,policy,available,visible,answer_from,correct,choose,gen):
    fit,gate,test=parts
    auxiliary=sorted(set(cases)-set(fit+gate+test))
    assert len(auxiliary)==16 and all('GoogleSearch' in actions(refs[k]) for k in auxiliary)
    assert len({tuple(c['images']) for c in cases.values()})==len(cases)==107
    keys=(gate+auxiliary)[int(os.environ['PART'])::4]
    call=visual_tools(original,visible,choose,gen)
    stored,_=fit_features(keys,context,call,policy,visible,answer_from,correct,choose,gen)
    save('private_feedback',dict(attempted=keys,records=stored))


if __name__=='__main__':
    main(locator=locate,experiment=collect)
