from skimage.util import crop
import numpy as np
from skimage.transform import rescale, rotate
from random import randrange, choice, uniform
from tqdm import tqdm
from utils import sliding_window, Rect

from params import BOX_SIZE, SCALES


def get_all_faces(imgs, labels, augment=True):
    '''Get a list of face images of size BOX_SIZE'''
    faces = []
    for img_id, rects in tqdm(labels.items(), desc='Extracting all faces'):
        for face_rect in rects:
            face = face_rect.extract_from_img(imgs[img_id])
            ratio = face.shape[1] / face.shape[0]
            
            if ratio < BOX_SIZE.w / BOX_SIZE.h:
                scaleW = BOX_SIZE.w / face.shape[1]
                face = rescale(face, scaleW, multichannel=False)
                # crop top and bottom
                diffH = face.shape[0] - BOX_SIZE.h
                face_cropped = crop(face, ((diffH // 2, diffH - diffH//2), (0, 0)))
            else:
                scaleH = BOX_SIZE.h / face.shape[0]
                face = rescale(face, scaleH, multichannel=False)
                # crop left and right
                diffW = face.shape[1] - BOX_SIZE.w
                face_cropped = crop(face, ((0, 0), (diffW // 2, diffW - diffW // 2)))
                
            faces.append(face_cropped.astype(np.float32))

    if augment:
        return augment_data(faces)
    else:
        return faces


def get_some_nonfaces(imgs, labels, num_examples):
    '''Get some random nonface images of size BOX_SIZE'''
    nonfaces = []
    pbar = tqdm(total=num_examples, desc='Extracting some nonfaces')
    while True:
        for img_id, img in imgs.items():
            # random scale
            scale = choice(SCALES)
            scaled_img = rescale(img, scale, multichannel=False)
            
            # random position
            try:
                y = randrange(scaled_img.shape[0] - BOX_SIZE.h)
                x = randrange(scaled_img.shape[1] - BOX_SIZE.w)
            except ValueError:  # images to small to extract a rect
                continue

            random_rect = Rect(x, y, BOX_SIZE.w, BOX_SIZE.h)
            
            if not random_rect.overlap(*labels[img_id], threshold=0.4):
                nonfaces.append(random_rect.extract_from_img(scaled_img))
                pbar.update()
                if len(nonfaces) >= num_examples:
                    return nonfaces


def get_all_nonfaces(train_imgs, labels):
    '''Get all nonfaces images from sliding window of size BOX_SIZE'''
    for id, img in tqdm(train_imgs.items(), desc='Extracting all nonfaces'):
        for scale in SCALES:
            img_scaled = rescale(img, scale, multichannel=False)
            true_rects = [r.scale(scale) for r in labels[id]]
            for rect, window in sliding_window(img_scaled):
                rect.scale(1 / scale)
                if not rect.overlap(*true_rects, threshold=0.4):
                    yield window
                   

def augment_data(imgs):
    '''From a list of face images, return a list of more faces (with flipped imgs)'''
    new_imgs = []
    for img in imgs:
        new_imgs.append(img)
        new_imgs.append(img[:, ::-1])  # horizontal flip

    return new_imgs


if __name__ == '__main__':
    from data_loading import load_train_imgs_and_labels

    train_imgs, labels = load_train_imgs_and_labels()

    # 1. Get all faces and some nonfaces then train the model
    train_faces = get_all_faces(train_imgs, labels)
    train_nonfaces = get_some_nonfaces(train_imgs, labels, len(train_faces) * 2)

