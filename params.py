from utils import Box

from skimage.feature import hog

DATA_DIR = './images'
BOX_SIZE = Box(h=100, w=66).scale(2)

WINDOW_STEP = 20
SCALES = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]

CV_FOLDS = 5


def features_extract_fn(img):
    return hog(img, orientations=16, pixels_per_cell=(32, 32),
               cells_per_block=(3, 3), block_norm='L2-Hys')
