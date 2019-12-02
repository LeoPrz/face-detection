from skimage.transform import rescale
from utils import sliding_window
from tqdm import tqdm
import numpy as np

from params import WINDOW_STEP, SCALES, features_extract_fn


def find_faces(model, imgs, threshold=None, step=WINDOW_STEP):
    '''Slide a window and find predictions that are over a threshold (or all if threshold is None)'''
    if threshold is None:
        threshold = -np.Inf

    rects_dict = {}
    scores_dict = {}
    for img_id, img in tqdm(imgs.items(), desc='Predicting faces'):
        rects = []
        windows = []
        for scale in SCALES:
            img_scaled = rescale(img, scale, multichannel=False)
            for rect, window in sliding_window(img_scaled, step=step):
                rect.scale(1 / scale)
                rects.append(rect)
                windows.append(window)

        rects = np.array(rects)
        features = []
        for window in tqdm(windows, desc='Computing features'):
            features.append(features_extract_fn(window))
        try:
            pred_proba = model.predict_proba(features)[:, 1]
        except AttributeError:
            pred_proba = model.decision_function(features)
           
        mask = pred_proba >= threshold

        rects_dict.update({img_id: rects[mask]})
        scores_dict.update({img_id: pred_proba[mask]})

    return remove_duplicate_rects(rects_dict, scores_dict)


def remove_duplicate_rects(rects_dict, scores_dict):
    '''Remove predictions that overlap each other and pick the ones with the better scores'''
    ret_rects = {}
    ret_scores = {}
    for i, rects in rects_dict.items():
        removed_rects = []
        scores = scores_dict[i]
        for rect1, score1 in zip(rects, scores):
            if rect1 in removed_rects:
                continue

            for rect2, score2 in zip(rects, scores):
                if rect2 in removed_rects or rect1 == rect2:
                    continue

                if rect1.overlap(rect2):
                    if score1 > score2 and rect2:
                        removed_rects.append(rect2)
                    elif score1 <= score2:
                        removed_rects.append(rect1)
        
        ret_scores.update({i: [s for idx, s in enumerate(scores) if rects[idx] not in removed_rects]})
        ret_rects.update({i: [r for r in rects if r not in removed_rects]})

    return ret_rects, ret_scores

