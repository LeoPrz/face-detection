import pickle
from inference import find_faces
from data_loading import load_test_imgs
from tqdm import tqdm

if __name__ == '__main__':
    model_path = 'model.pickle'
    model = pickle.load(open(model_path, 'rb'))
    imgs = load_test_imgs()

    faces, scores = find_faces(model, imgs, threshold=-0.1)
    with open('detection.txt', 'w') as f:
        for i, rects in tqdm(faces.items(), desc='Writing to file'):
            for r, s in zip(rects, scores[i]):
                str = f'{int(i)} {int(r.y)} {int(r.x)} {int(r.h)} {int(r.w)} {s}\n'
                f.write(str)
