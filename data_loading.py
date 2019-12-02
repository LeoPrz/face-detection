import numpy as np
from skimage.io import imread
from os.path import join, basename, splitext
from tqdm import tqdm as tqdm
from glob import glob
from utils import Rect

from params import DATA_DIR


def labels_to_dict(labels):
    '''ndarray of labels (Nx5) to dict of Rect lists'''
    labels_dict = {k: [] for k in np.unique(labels[:, 0])}
    for k, i, j, h, l in labels:
        labels_dict[k].append(Rect(j, i, l, h))
    return labels_dict


def load_imgs(dir, as_grey=True):
    '''Load all jpg in the directory and returns a dict'''
    imgs = {}
    imgs_names = glob(join(dir, '*.jpg'))
    for f in tqdm(imgs_names, desc='Loading images'):
        img = imread(f, as_grey)
        img_id = int(splitext(basename(f))[0])
        imgs.update({img_id: img.astype(np.float32)})
    
    return imgs


def load_test_imgs(data_dir=DATA_DIR, as_grey=True):
    return load_imgs(join(data_dir, 'test'), as_grey)


def load_train_imgs_and_labels(data_dir=DATA_DIR, as_grey=True):
    labels_nd = np.loadtxt(join(data_dir, 'label.txt'), dtype=np.int)
    labels = labels_to_dict(labels_nd)
    return load_imgs(join(data_dir, 'train'), as_grey), labels


if __name__ == '__main__':
    train_imgs, labels = load_train_imgs_and_labels()

    assert len(train_imgs) == len(labels)
    assert all(isinstance(i, list) for i in labels.values())
    assert all(isinstance(i, np.ndarray) for i in train_imgs.values())

    print('\nSummary of available data:')
    print(f'\t#images: {len(train_imgs)}')
    print(f'\t#faces: {sum(len(i) for i in labels.values())}')
