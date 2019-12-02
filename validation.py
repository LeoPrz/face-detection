from copy import deepcopy
import numpy as np
from tqdm import tqdm
from face_extraction import get_all_faces, get_all_nonfaces
from training import get_features_and_labels
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def assign_faces(pred_rects, label_rects):
    '''For each image, get a list of 1s and 0s: 1 for the rects that correspond to a face'''
    results = {}
    pred_rects = deepcopy(pred_rects)

    for i, rects in pred_rects.items():
        assigned_rects = []
        for face in label_rects[i]:
            rects_on_face = [r for r in rects if r not in assigned_rects and r.overlap(face)]
            overlap_score = [r.inter_over_union(face) for r in rects_on_face]

            if overlap_score:
                assigned_rects.append(rects_on_face[np.argmax(overlap_score)])

        results.update({i: [1 if r in assigned_rects else 0 for r in rects]})

    return results


def filter_faces(pred_rects, label_rects):
    '''Return the rects dict with only rects that correspond to a face'''
    found_labels = assign_faces(pred_rects, label_rects)
    return {i: [r for ri, r in enumerate(pred_rects[i]) if found_labels[i][ri]] for i in pred_rects}


def get_precision_recall_f1(model, val_imgs, val_labels):
    val_faces = get_all_faces(val_imgs, val_labels)
    val_nonfaces = get_all_nonfaces(val_imgs, val_labels)
    val_feats, val_binary_labels = get_features_and_labels(val_faces, val_nonfaces)
    try:
        pred_proba = model.predict_proba(val_feats)[:, 1]
    except AttributeError:
        pred_proba = model.decision_function(val_feats)

    # PR curve
    p, r, t = precision_recall_curve(val_binary_labels, pred_proba)
    p = np.array(p)
    r = np.array(r)
    t = np.array(t)
    f1 = 2*p*r/(p+r)

    return p, r, f1, t


