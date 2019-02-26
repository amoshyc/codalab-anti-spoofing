import json
from pathlib import Path

import dlib
import numpy as np
from PIL import Image
from tqdm import tqdm

import skimage
from skimage import filters


class ImageResizer:
    def __init__(self, img_dir):
        self.idx = 0
        self.img_dir = Path(img_dir)

    def __call__(self, rgb_path, dep_path, ifr_path):
        dst_rgb_path = self.img_dir / f'{self.idx:08d}.rgb.jpg'
        dst_dep_path = self.img_dir / f'{self.idx:08d}.dep.jpg'
        dst_ifr_path = self.img_dir / f'{self.idx:08d}.ifr.jpg'

        rgb_img = Image.open(rgb_path).resize((96, 96)).convert('RGB')
        dep_img = Image.open(dep_path).resize((96, 96)).convert('L')
        ifr_img = Image.open(ifr_path).resize((96, 96)).convert('L')

        rgb_img.save(dst_rgb_path)
        dep_img.save(dst_dep_path)
        ifr_img.save(dst_ifr_path)

        self.idx += 1

        return {
            'rgb_path': str(dst_rgb_path),
            'dep_path': str(dst_dep_path),
            'ifr_path': str(dst_ifr_path),
        }

class KptPredictor:
    def __init__(self, dat_path):
        self.predictor = dlib.shape_predictor(dat_path)

    def __call__(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.uint8(img)
        h, w = img.shape[:2]
        rect = dlib.rectangle(0, 0, w, h)
        result = self.predictor(img, rect)
        kpt = np.float32([(p.x, p.y) for p in result.parts()])
        kpt = kpt / np.float32([w, h])
        return {
            'kpt': kpt.tolist()
        }

class MetaExtractor:
    def __init__(self):
        pass
    
    def __call__(self, img_path, label):
        return {
            'label': int(label),
            'subject': img_path.parents[2].stem[-4:]
        }


# Train
with open('../phase1/train_list.txt') as f:
    data = [line.split() for line in f.readlines()]
anns = []
src_dir = Path('../phase1/')
dst_dir = Path('./data/train/')
dst_dir.mkdir(parents=True, exist_ok=True)
img_resizer = ImageResizer(dst_dir)
kpt_predictor = KptPredictor('./shape_predictor_68_face_landmarks.dat')
meta_extractor = MetaExtractor()
for (rgb_path, dep_path, ifr_path, label) in tqdm(data):
    rgb_path = src_dir / rgb_path
    dep_path = src_dir / dep_path
    ifr_path = src_dir / ifr_path
    anns.append({
        # **img_resizer(rgb_path, dep_path, ifr_path),
        'rgb_path': str(rgb_path),
        'dep_path': str(dep_path),
        'ifr_path': str(ifr_path),
        **kpt_predictor(rgb_path),
        **meta_extractor(rgb_path, label),
    })
with (dst_dir / 'anns.json').open('w') as f:
    json.dump(anns, f, indent=2, ensure_ascii=False)


# Valid
with open('../phase1/val_public_list.txt') as f:
    data = [line.split() for line in f.readlines()]
anns = []
src_dir = Path('../phase1/')
dst_dir = Path('./data/valid/')
dst_dir.mkdir(parents=True, exist_ok=True)
img_resizer = ImageResizer(dst_dir)
kpt_predictor = KptPredictor('./shape_predictor_68_face_landmarks.dat')
for (rgb_path, dep_path, ifr_path) in tqdm(data):
    rgb_path = src_dir / rgb_path
    dep_path = src_dir / dep_path
    ifr_path = src_dir / ifr_path
    anns.append({
        'rgb_path': str(rgb_path),
        'dep_path': str(dep_path),
        'ifr_path': str(ifr_path),
        # **img_resizer(rgb_path, dep_path, ifr_path),
        **kpt_predictor(rgb_path),
    })
with (dst_dir / 'anns.json').open('w') as f:
    json.dump(anns, f, indent=2, ensure_ascii=False)