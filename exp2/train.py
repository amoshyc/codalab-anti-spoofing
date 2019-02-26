import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib as mpl
mpl.use('SVG')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.transforms import functional as tf

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

import util
from dataset import CASIASURFTrain
from model import CASIASURFModel

train_set = CASIASURFTrain('./data/train/train_anns.json', shuffle=False, augment=True)
valid_set = CASIASURFTrain('./data/train/valid_anns.json', shuffle=False, augment=False)
train_loader = DataLoader(train_set, 64, shuffle=True, num_workers=3)
valid_loader = DataLoader(valid_set, 64, shuffle=True, num_workers=3)

device = 'cuda'
model = CASIASURFModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)
print(log_dir)


def train(pbar):
    model.train()
    data = {
        'loss': util.RunningMean(),
        'true': util.RunningData(),
        'pred': util.RunningData(),
    }
    for img_b, lbl_b in iter(train_loader):
        img_b = img_b.to(device)
        lbl_b = lbl_b.to(device)

        optimizer.zero_grad()
        out_b = model(img_b)
        loss = criterion(out_b, lbl_b)
        loss.backward()
        optimizer.step()

        data['loss'].update(loss.detach().item())
        data['true'].update(lbl_b.detach().cpu())
        data['pred'].update(out_b.detach().cpu())
        pbar.set_postfix(loss=data['loss'])
        pbar.update(img_b.size(0))
    return data


def valid(pbar):
    model.eval()
    data = {
        'loss': util.RunningMean(),
        'true': util.RunningData(),
        'pred': util.RunningData(),
    }
    for img_b, lbl_b in iter(valid_loader):
        img_b = img_b.to(device)
        lbl_b = lbl_b.to(device)
        out_b = model(img_b)
        loss = criterion(out_b, lbl_b)
        data['loss'].update(loss.detach().item())
        data['true'].update(lbl_b.detach().cpu())
        data['pred'].update(out_b.detach().cpu())
        pbar.set_postfix(loss=data['loss'])
        pbar.update(img_b.size(0))
    return data


def log(epoch, train_data, valid_data):
    # ROC Curve of each epoch
    fig, ax = plt.subplots(1, 2, dpi=100, figsize=(12, 5))
    train_metric = util.roc(ax[0],
        train_data['true'].value().numpy(), 
        train_data['pred'].value().numpy())
    ax[0].set_title('Train ROC ({:.5f})'.format(train_metric['auc']))
    valid_metric = util.roc(ax[1],
        valid_data['true'].value().numpy(),
        valid_data['pred'].value().numpy())
    ax[1].set_title('Valid ROC ({:.5f})'.format(valid_metric['auc']))
    fig.savefig(log_dir / f'{epoch:03d}_roc.svg')
    plt.close()

    # Hist
    fig, ax = plt.subplots()
    ax.hist(valid_data['pred'].value().numpy(), bins=50)
    fig.savefig(log_dir / f'{epoch:03d}_hist.svg')
    plt.close()

    # Metric
    csv_path = log_dir / 'metric.csv'
    df = pd.read_csv(csv_path) if epoch > 0 else pd.DataFrame()
    df = df.append({
        'epoch': epoch,
        'train_loss': train_data['loss'].value(),
        'valid_loss': valid_data['loss'].value(),
        **{f'train_{k}': v for k, v in train_metric.items()},
        **{f'valid_{k}': v for k, v in valid_metric.items()},
    }, ignore_index=True)
    df.to_csv(csv_path, index=False)

    # Metric Curve
    fig, ax = plt.subplots(3, 3, dpi=100, figsize=(20, 20))
    df[['train_loss', 'valid_loss']].plot(kind='line', ax=ax[0, 0])
    df[['train_acc', 'valid_acc']].plot(kind='line', ax=ax[0, 1])
    df[['train_auc', 'valid_auc']].plot(kind='line', ax=ax[0, 2])
    df[['train_tpr@1e-4', 'valid_tpr@1e-4']].plot(kind='line', ax=ax[1, 0])
    df[['train_tpr@1e-3', 'valid_tpr@1e-3']].plot(kind='line', ax=ax[1, 1])
    df[['train_tpr@1e-2', 'valid_tpr@1e-2']].plot(kind='line', ax=ax[1, 2])
    df[['train_acer', 'valid_acer']].plot(kind='line', ax=ax[2, 0])
    df[['train_apcer', 'valid_apcer']].plot(kind='line', ax=ax[2, 1])
    df[['train_npcer', 'valid_npcer']].plot(kind='line', ax=ax[2, 2])
    fig.savefig(log_dir / 'metric.svg')
    plt.close()

    # Model
    if df['valid_tpr@1e-4'].idxmax() == epoch:
        print('  Dump weight', flush=True)
        torch.save(model.state_dict(), log_dir / 'model.pth')


for epoch in range(30):
    print('Epoch', epoch, flush=True)

    with tqdm(total=len(train_set), desc='  Train') as pbar:
        train_data = train(pbar)

    with torch.no_grad():
        with tqdm(total=len(valid_set), desc='  Valid') as pbar:
            valid_data = valid(pbar)
        
        log(epoch, train_data, valid_data)

