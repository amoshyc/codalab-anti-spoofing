import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as tf


class CASIASURFTrain(Dataset):
    def __init__(self, ann_path, shuffle=False, augment=False):
        super().__init__()
        self.augment = augment
        ann_path = Path(ann_path)
        with ann_path.open() as f:
            self.anns = json.load(f)
        if shuffle:
            random.shuffle(self.anns)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        
        rgb_img = Image.open(ann['rgb_path']).convert('RGB')
        dep_img = Image.open(ann['dep_path']).convert('L')
        ifr_img = Image.open(ann['ifr_path']).convert('L')

        if self.augment:
            rgb_img, dep_img, ifr_img = \
                self.transform(rgb_img, dep_img, ifr_img)

        rgb_img = tf.to_tensor(rgb_img) # [3, 64, 64]
        dep_img = tf.to_tensor(dep_img) # [1, 64, 64]
        ifr_img = tf.to_tensor(ifr_img) # [1, 64, 64]

        img = torch.cat((rgb_img, dep_img, ifr_img), dim=0) # [5, 64, 64]
        lbl = torch.FloatTensor([ann['lbl']])

        return img, lbl

    def transform(self, rgb_img, dep_img, ifr_img):
        if random.randint(0, 1) == 0:
            rgb_img = tf.hflip(rgb_img)
            dep_img = tf.hflip(dep_img)
            ifr_img = tf.hflip(ifr_img)
        
        if random.randint(0, 1) == 0:
            angle = random.randint(-15, +15)
            offset = random.randint(-10, +10)
            scale = random.uniform(0.85, 1.15)
            rgb_img = tf.affine(rgb_img, angle, (offset, offset), scale, 0.0)
            dep_img = tf.affine(dep_img, angle, (offset, offset), scale, 0.0)
            ifr_img = tf.affine(ifr_img, angle, (offset, offset), scale, 0.0)

        return rgb_img, dep_img, ifr_img


class CASIASURFInfer(Dataset):
    def __init__(self, txt_path):
        super().__init__()
        txt_path = Path(txt_path)
        self.root_dir = txt_path.parent
        with txt_path.open() as f:
            self.paths = [line.split() for line in f.readlines()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rgb_path, dep_path, ifr_path = self.paths[idx]
        rgb_img = Image.open(self.root_dir / rgb_path).convert('RGB')
        dep_img = Image.open(self.root_dir / dep_path).convert('L')
        ifr_img = Image.open(self.root_dir / ifr_path).convert('L')

        rgb_img = rgb_img.resize((64, 64))
        dep_img = dep_img.resize((64, 64))
        ifr_img = ifr_img.resize((64, 64))

        rgb_img = tf.to_tensor(rgb_img) # [3, 64, 64]
        dep_img = tf.to_tensor(dep_img) # [1, 64, 64]
        ifr_img = tf.to_tensor(ifr_img) # [1, 64, 64]

        img = torch.cat((rgb_img, dep_img, ifr_img), dim=0) # [5, 64, 64]
        return img


if __name__ == '__main__':
    train_set = CASIASURFTrain('./data/train/train_anns.json')
    valid_set = CASIASURFTrain('./data/train/valid_anns.json')
    print(len(train_set))
    print(len(valid_set))

    img, lbl = train_set[-1]
    print(img.size())
    print(lbl.size())
    img, lbl = valid_set[-1]
    print(img.size())
    print(lbl.size())

    infer_set = CASIASURFInfer('../phase1/val_public_list.txt')
    print(len(infer_set))
    print(infer_set[-1].size())
