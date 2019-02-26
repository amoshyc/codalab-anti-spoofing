import json
import random
from pathlib import Path
from itertools import islice
from datetime import datetime
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader, ConcatDataset, Subset

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

mpl.use('SVG')
plt.style.use('seaborn')

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

import util
import config
from model import CASIASURFModel
from dataset import CASIASURFData

CYCLE_STEP = 200 # number of training step per cycle
BATCH_SIZE = 64

train_set = CASIASURFData('./data/train/anns.json', config.train_subjects, mode='train')
tnval_set = CASIASURFData('./data/train/anns.json', config.tnval_subjects, mode='infer')
valid_set = CASIASURFData('./data/valid/anns.json', mode='infer')
visul_set = ConcatDataset([
    Subset(train_set, random.sample(range(len(train_set)), 50)),
    Subset(tnval_set, random.sample(range(len(tnval_set)), 50)),
    Subset(valid_set, random.sample(range(len(valid_set)), 50)),
])
train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=4)
tnval_loader = DataLoader(tnval_set, BATCH_SIZE, False, num_workers=4)
visul_loader = DataLoader(visul_set, 10, False, num_workers=2)

device = torch.device('cuda')
model = CASIASURFModel().to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)
print(log_dir)


def train(pbar):
    model.train()
    state = defaultdict(list)
    for inp_b, lbl_b in islice(iter(train_loader), CYCLE_STEP):
        inp_b = inp_b.to(device)
        lbl_b = lbl_b.to(device)

        optimizer.zero_grad()
        out_b = model(inp_b)
        loss = criterion(out_b, lbl_b)
        loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        pbar.update(inp_b.size(0))
        pbar.set_postfix(loss='{:.5f}'.format(loss))
        state['loss'].append(loss)
        state['true'].append(lbl_b.detach().cpu())
        state['pred'].append(torch.sigmoid(out_b.detach().cpu()))
    pbar.set_postfix()
    return state


def tnval(pbar):
    model.eval()
    state = defaultdict(list)
    for inp_b, lbl_b in iter(tnval_loader):
        inp_b = inp_b.to(device)
        lbl_b = lbl_b.to(device)
        out_b = model(inp_b)

        loss = criterion(out_b, lbl_b).item()
        pbar.update(inp_b.size(0))
        pbar.set_postfix(loss='{:.5f}'.format(loss))
        state['loss'].append(loss)
        state['true'].append(lbl_b.detach().cpu())
        state['pred'].append(torch.sigmoid(out_b.detach().cpu()))
    pbar.set_postfix()
    return state


def visul(pbar, cycle):
    model.eval()
    cycle_dir = log_dir / f'{cycle:03d}'
    cycle_dir.mkdir(exist_ok=True)

    for i, (inp_b, lbl_b) in enumerate(iter(visul_loader)):
        N, _, H, W = inp_b.size()

        with util.GradCam(model, [model.features]) as gcam:
            out_b = gcam(inp_b.to(device))
            out_b.backward(torch.full_like(out_b, +1.0))
            gcam_pos_b = gcam.get(model.features)
            gcam_pos_b = F.interpolate(gcam_pos_b, [H, W], mode='bilinear', align_corners=False)
        
        with util.GradCam(model, [model.features]) as gcam:
            out_b = gcam(inp_b.to(device))
            out_b.backward(torch.full_like(out_b, -1.0))
            gcam_neg_b = gcam.get(model.features)
            gcam_neg_b = F.interpolate(gcam_neg_b, [H, W], mode='bilinear', align_corners=False)
        
        with util.GuidedBackPropogation(model) as gdbp:
            inp_rg_b = inp_b.to(device).requires_grad_()
            out_b = gdbp(inp_rg_b)
            out_b.backward(torch.full_like(out_b, +1.0))
            grad_pos_b = gdbp.get(inp_rg_b)
            grad_pos_b = grad_pos_b.mean(dim=1, keepdim=True)

        with util.GuidedBackPropogation(model) as gdbp:
            inp_rg_b = inp_b.to(device).requires_grad_()
            out_b = gdbp(inp_rg_b)
            out_b.backward(torch.full_like(out_b, -1.0))
            grad_neg_b = gdbp.get(inp_rg_b)
            grad_neg_b = grad_neg_b.mean(dim=1, keepdim=True)

        mixed_pos_b = gcam_pos_b * grad_pos_b
        mixed_neg_b = gcam_neg_b * grad_neg_b

        out_b = model(inp_b.to(device)).cpu()
        lbl_b = lbl_b.view(N, 1, 1, 1).expand(N, 3, H, W)
        out_b = out_b.view(N, 1, 1, 1).expand(N, 3, H, W)

        save_image(torch.cat([
            inp_b[:, 0:3], # rgb
            util.colorize(inp_b[:, 3:4], plt.cm.gray), # dep
            util.colorize(inp_b[:, 4:5], plt.cm.gray), # ifr
            util.colorize(util.normalize(gcam_pos_b)),
            util.colorize(util.normalize(grad_pos_b)),
            util.colorize(util.normalize(mixed_pos_b)),
            lbl_b,
            out_b,
            util.colorize(util.normalize(gcam_neg_b)),
            util.colorize(util.normalize(grad_neg_b)),
            util.colorize(util.normalize(mixed_neg_b)),
        ], dim=0), cycle_dir / f'{i:03d}.png', nrow=N)

        pbar.update(inp_b.size(0))


def log(cycle, train_state, tnval_state):
    for state in [train_state, tnval_state]:
        state['loss'] = torch.FloatTensor(state['loss']).mean().item()
        state['pred'] = torch.cat(state['pred'], dim=0).numpy().ravel()
        state['true'] = torch.cat(state['true'], dim=0).numpy().ravel()
    
    train_metric = {
        'loss': train_state['loss'],
        **util.roc_metric(train_state['true'], train_state['pred']),
        **util.cls_metric(train_state['true'], train_state['pred'])
    }
    tnval_metric = {
        'loss': tnval_state['loss'],
        **util.roc_metric(tnval_state['true'], tnval_state['pred']),
        **util.cls_metric(tnval_state['true'], tnval_state['pred']),
    }
    print('  train_loss:', train_metric['loss'], flush=True)
    print('  tnval_loss:', tnval_metric['loss'], flush=True)

    json_path = log_dir / 'metric.json'
    if cycle > 0:
        df = pd.read_json(json_path, orient='records')
    else:
        df = pd.DataFrame()
    df = df.append({
        'cycle': cycle,
        **{f'train_{k}': v for k, v in train_metric.items()},
        **{f'tnval_{k}': v for k, v in tnval_metric.items()},
    }, ignore_index=True)
    with json_path.open('w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(3, 3, dpi=100, figsize=(20, 20))
    df[['train_loss', 'tnval_loss']].plot(ax=ax[0, 0])
    df[['train_acc', 'tnval_acc']].plot(ax=ax[0, 1])
    df[['train_auc', 'tnval_auc']].plot(ax=ax[0, 2])
    df[['train_tpr@fpr=1e-4', 'tnval_tpr@fpr=1e-4']].plot(ax=ax[1, 0])
    df[['train_tpr@fpr=1e-3', 'tnval_tpr@fpr=1e-3']].plot(ax=ax[1, 1])
    df[['train_tpr@fpr=1e-2', 'tnval_tpr@fpr=1e-2']].plot(ax=ax[1, 2])
    df[['train_acer', 'tnval_acer']].plot(ax=ax[2, 0])
    df[['train_apcer', 'tnval_apcer']].plot(ax=ax[2, 1])
    df[['train_npcer', 'tnval_npcer']].plot(ax=ax[2, 2])
    fig.savefig(log_dir / 'metric.svg')
    plt.close()

    cycle_dir = log_dir / f'{cycle:03d}'
    cycle_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(tnval_state['pred'], bins=50)
    fig.savefig(cycle_dir / 'hist.svg')
    plt.close()

    path = cycle_dir / f'weight.pth'
    torch.save(model.state_dict(), path)
    cycle_id = df['tnval_tpr@fpr=1e-4'].idxmax()
    best_val = df['tnval_tpr@fpr=1e-4'].max()
    print('  Best: {:03d} {:.5f}'.format(cycle_id, best_val), flush=True)


for cycle in range(30):
    print('Cycle', cycle, flush=True)
    
    # scheduler.step()
    
    with tqdm(total=CYCLE_STEP * BATCH_SIZE, desc='  Train', ascii=True) as pbar:
        train_state = train(pbar)
    with tqdm(total=len(visul_set), desc='  Visul', ascii=True) as pbar:
        visul(pbar, cycle)
        
    with torch.no_grad():
        with tqdm(total=len(tnval_set), desc='  Tnval', ascii=True) as pbar:
            tnval_state = tnval(pbar)
        log(cycle, train_state, tnval_state)
    