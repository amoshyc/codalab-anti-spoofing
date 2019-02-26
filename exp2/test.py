from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

import matplotlib as mpl
mpl.use('SVG')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

seed = 999
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from model import CASIASURFModel
from dataset import CASIASURFInfer

txt_path = Path('../phase1/val_public_list.txt')
infer_set = CASIASURFInfer(txt_path)
infer_loader = DataLoader(infer_set, 64, shuffle=False, num_workers=3)

device = 'cuda'
model = CASIASURFModel().to(device)
model.load_state_dict(torch.load('log/2019.02.06-22:42:49/model.pth'))
model.eval()

pred_dir = Path('./infer/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
pred_dir.mkdir(parents=True, exist_ok=True)
print(pred_dir)

def mean_pred(pbar, loader):
    pred = []
    for img_b in iter(loader):
        img_b = img_b.to(device)
        act_b = model.features(img_b)

        pred.extend(out_b.numpy().ravel().tolist())
        pbar.update(len(img_b))
    return pred


with torch.no_grad():
    with tqdm(total=len(infer_set)) as pbar:
        pred = infer(pbar)
    
    mean = torch.zeros(64, 64)
    with tqdm(total=)