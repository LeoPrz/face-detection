import numpy as np
from tqdm import tqdm
from sklearn.model_selection import cross_validate

from face_extraction import get_all_nonfaces, get_all_faces
from params import features_extract_fn, CV_FOLDS


def get_features_and_labels(faces, nonfaces):
    faces_feat = list(map(features_extract_fn, faces))
    nonfaces_feat = list(map(features_extract_fn, nonfaces))
    binary_labels = [1] * len(faces_feat) + [0] * len(nonfaces_feat)
    return faces_feat + nonfaces_feat, binary_labels


def cross_val_model(model, faces, nonfaces):
    feats, labels = get_features_and_labels(faces, nonfaces)
    results = cross_validate(model, feats, labels, cv=CV_FOLDS, n_jobs=-1,
                             scoring=['balanced_accuracy', 'precision', 'recall', 'f1'],
                             return_train_score=False)
    return {k: l.mean() for k, l in results.items()}


def retrain_with_false_positives(imgs, labels, model, previous_feats, previous_labels):
    nonface_feats = [features_extract_fn(x) for x in get_all_nonfaces(imgs, labels)]
    preds = model.predict(nonface_feats)
    false_pos = list(np.array(nonface_feats)[preds == 1])
    print(' Number of false positives:', len(false_pos))
    all_feats = previous_feats + false_pos
    feat_labels = previous_labels + [0] * len(false_pos)
    model.fit(all_feats, feat_labels)


if __name__ == '__main__':
    from sklearn.svm import LinearSVC

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve

    import matplotlib.pyplot as plt
    from pprint import pprint
    import pickle

    from face_extraction import get_some_nonfaces
    from data_loading import load_train_imgs_and_labels
    from validation import get_precision_recall_f1
    from visualization import plot_precision_recall_f1

    all_imgs, all_labels = load_train_imgs_and_labels()
    clf = LinearSVC(C=1)

    final_training = False

    # cross val or not
    if 0:
        print('\n' * 2 + '-' * 50)
        print('\tCross-validation 5 folds')
        print('\tWith all faces and some nonfaces')
        print('-' * 50 + '\n')

        train_faces = get_all_faces(all_imgs, all_labels)
        train_nonfaces = get_some_nonfaces(all_imgs, all_labels,
                                           num_examples=len(train_faces) * 3)

        cv_results = cross_val_model(clf, train_faces, train_nonfaces)
        print('Results of CV:')
        pprint(cv_results)

    if not final_training:
        # hacky way to split dicts
        train_imgs, val_imgs, train_labels, val_labels = [
            dict(x) for x in train_test_split(list(sorted(all_imgs.items())),
                                              list(sorted(all_labels.items())),
                                              test_size=0.1)
        ]
    else:
        train_imgs, train_labels = all_imgs, all_labels

    print('\nExtracting training data and computing features')
    train_faces = get_all_faces(train_imgs, train_labels)
    train_nonfaces = get_some_nonfaces(train_imgs, train_labels, num_examples=len(train_faces) * 3)
    feats, binary_labels = get_features_and_labels(train_faces, train_nonfaces)

    print('\nTraining the model')
    clf.fit(feats, binary_labels)

    print('\nRetraining with false positives')
    retrain_with_false_positives(train_imgs, train_labels, clf, feats, binary_labels)

    if not final_training:
        print('\nEvaluation of the model (phase 2)')
        p, r, f, t = get_precision_recall_f1(clf, val_imgs, val_labels)
        plot_precision_recall_f1(p, r, f, t)

    with open('model.pickle', 'wb') as f:
        pickle.dump(clf, f)

