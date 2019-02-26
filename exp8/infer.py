from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
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
from dataset import CASIASURFData

txt_path = Path('../phase1/val_public_list.txt')
valid_set = CASIASURFData('./data/valid/anns.json', mode='infer')
valid_loader = DataLoader(valid_set, 64, shuffle=False, num_workers=3)

device = 'cuda'
model = CASIASURFModel().to(device)
model.load_state_dict(torch.load('log/2019.02.26-14:28:00/007/weight.pth'))
model.eval()

pred_dir = Path('./infer/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
pred_dir.mkdir(parents=True, exist_ok=True)
print(pred_dir)

def infer(pbar):
    pred = []
    for inp_b, _ in iter(valid_loader):
        out_b = torch.sigmoid(model(inp_b.to(device))).cpu()
        pred.extend(out_b.numpy().ravel().tolist())
        pbar.update(len(inp_b))
    return pred


with torch.no_grad():
    with tqdm(total=len(valid_set)) as pbar:
        pred = infer(pbar)
    
    with txt_path.open() as f:
        lines = f.readlines()

    df = pd.read_csv(txt_path, sep=' ', header=None)
    df[3] = pred
    df.to_csv(pred_dir / 'pred.txt', index=None, header=None, sep=' ', float_format='%.7f')

    fig, ax = plt.subplots()
    ax.hist(np.float32(pred), bins=50)
    fig.savefig(str(pred_dir / 'hist.svg'))
    plt.close()