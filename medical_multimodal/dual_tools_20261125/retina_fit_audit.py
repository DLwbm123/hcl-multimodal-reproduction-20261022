"""Post-hoc analysis of the feedback images only; do different specialists actually help?"""
import numpy as np
from retina import component, features, ratio, save, source, split, target, train

tool_train, fit, _, _ = split()
forest = train(tool_train)
errors = []
for k in fit:
    truth = target(k)
    target_ratio = ratio(truth > 0, truth == 2)
    row = []
    for model, ff in zip(forest, features(source(k))[0]):
        region = ff[:,:,2] < 2.5
        mask = np.zeros(region.shape, 'uint8')
        mask[region] = model.predict(ff[region])
        disc = component(mask > 0)
        cup = component((mask == 2) & disc)
        row.append(abs(ratio(disc,cup) - target_ratio))
    errors.append(row)
a = np.array(errors)
save('fit_audit',dict(n=len(fit),color_better=int(sum(a[:,0]<a[:,1])),
                      contrast_better=int(sum(a[:,1]<a[:,0])),ties=int(sum(a[:,0]==a[:,1])),
                      color_MAE=a[:,0].mean(),contrast_MAE=a[:,1].mean(),oracle_MAE=a.min(axis=1).mean()))
