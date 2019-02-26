import json
import random
from PIL import Image
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as tf

import util
import numpy as np
from skimage import morphology


class CASIASURFData(Dataset):
    def __init__(self, ann_path, subjects=None, mode='train'):
        super().__init__()
        self.mode = mode
        with Path(ann_path).open() as f:
            self.anns = json.load(f)
        if subjects:
            subjects = set(subjects)
            self.anns = [ann for ann in self.anns if ann['subject'] in subjects]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        
        rgb_img = Image.open(ann['rgb_path']).convert('RGB').resize((64, 64))
        dep_img = Image.open(ann['dep_path']).convert('L').resize((64, 64))
        ifr_img = Image.open(ann['ifr_path']).convert('L').resize((64, 64))

        inp = np.concatenate([
            tf.to_tensor(rgb_img).numpy(), # [3, H, W]
            tf.to_tensor(dep_img).numpy(), # [1, H, W]
            tf.to_tensor(ifr_img).numpy(), # [1, H, W]
        ], axis=0)
        for i in range(5):
            inp[i] = morphology.opening(inp[i], morphology.square(5))

        inp = torch.from_numpy(inp).float()
        lbl = torch.tensor([ann.get('label', 0.5)]).float()
        
        return inp, lbl


if __name__ == '__main__':
    import config
    train_set = CASIASURFData('./data/train/anns.json', config.train_subjects)
    tnval_set = CASIASURFData('./data/train/anns.json', config.tnval_subjects)
    for ds in [train_set, tnval_set]:
        print(len(ds))
        inp, lbl = ds[-1]
        print(inp.size())
        print(lbl.size())
        print('-' * 10)