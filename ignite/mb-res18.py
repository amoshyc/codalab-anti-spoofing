import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage import morphology
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('SVG')
plt.style.use('seaborn')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.utils import save_image
from torchvision.transforms import functional as tf
from torch.utils.data import DataLoader, Subset, ConcatDataset

import ignite
from ignite.engine import Engine, Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import RunningAverage, Loss, Accuracy
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

import util


class Data:
    def __init__(self, img_dir, txt_path):
        self.img_dir = Path(img_dir)
        with open(txt_path) as f:
            self.anns = [line.split() for line in f.readlines()]
        
    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]

        rgb_path, dep_path, ifr_path = ann[:3]
        rgb_path = self.img_dir / rgb_path
        dep_path = self.img_dir / dep_path
        ifr_path = self.img_dir / ifr_path
        lbl = float(ann[-1]) if len(ann) == 4 else 0.5

        rgb_img = Image.open(rgb_path).convert('RGB').resize((64, 64))
        dep_img = Image.open(dep_path).convert('L').resize((64, 64))
        ifr_img = Image.open(ifr_path).convert('L').resize((64, 64))

        inp = torch.cat([
            tf.to_tensor(rgb_img), 
            tf.to_tensor(dep_img), 
            tf.to_tensor(ifr_img)], dim=0).numpy()
        for i in range(5):
            inp[i] = morphology.opening(inp[i], morphology.square(5))

        inp = torch.from_numpy(inp).float()
        lbl = torch.tensor([lbl]).float()
        return inp, lbl


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_cnn = self.resnet18(3)
        self.dep_cnn = self.resnet18(1)
        self.ifr_cnn = self.resnet18(1)
        self.fuse = nn.Sequential(
            nn.Conv2d(512 * 3, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, inp_b):
        fmp_rgb_b = self.rgb_cnn(inp_b[:, 0:3])
        fmp_dep_b = self.dep_cnn(inp_b[:, 3:4])
        fmp_ifr_b = self.ifr_cnn(inp_b[:, 4:5])
        fmp_b = torch.cat([fmp_rgb_b, fmp_dep_b, fmp_ifr_b], dim=1)
        out_b = self.fuse(fmp_b)
        return out_b

    def resnet18(self, nc):
        net = resnet18(pretrained=False)
        return nn.Sequential(
            nn.Conv2d(nc, 64, (7, 7), padding=3),
            net.bn1,
            net.relu,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4
        )
        

def train():
    train_set = Data('../raw/', '../raw/train_list.txt')
    valid_set = Data('../raw/', '../raw/val_private_list.txt')
    visul_set = ConcatDataset([
        Subset(train_set, random.sample(range(len(train_set)), 50)),
        Subset(valid_set, random.sample(range(len(valid_set)), 50)),
    ])
    train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, 32, shuffle=True, num_workers=4)
    visul_loader = DataLoader(visul_set, 25, shuffle=True, num_workers=4)

    device = torch.device('cuda')
    model = Model()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
    log_dir.mkdir(parents=True)
    print(log_dir)
    train_history = defaultdict(list)
    valid_history = defaultdict(list)

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    inferer = create_supervised_evaluator(model, {}, device)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    ProgressBar(True).attach(trainer, ['loss'])

    metric = util.MyMetric(criterion)
    metric.attach(inferer, 'metric')

    ckpt = ModelCheckpoint(str(log_dir), 'dense121', save_interval=1, n_saved=50)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt, {
        'model': model
    })

    @trainer.on(Events.EPOCH_COMPLETED)
    def log(engine):
        inferer.run(train_loader, max_epochs=1)
        train_metrics = inferer.state.metrics['metric']
        print('Train: ', train_metrics, flush=True)

        inferer.run(valid_loader, max_epochs=1)
        valid_metrics = inferer.state.metrics['metric']
        print('Valid: ', valid_metrics, flush=True)

        for key in train_metrics.keys():
            train_history[key].append(train_metrics[key])
            valid_history[key].append(valid_metrics[key])
        
        fig, ax = plt.subplots(3, 3, dpi=100, figsize=(20, 20))
        xs = range(1, trainer.state.epoch + 1)
        for i, key in enumerate(train_metrics.keys()):
            ax[divmod(i, 3)].set_title(key.upper())
            ax[divmod(i, 3)].plot(xs, train_history[key], label='Train')
            ax[divmod(i, 3)].plot(xs, valid_history[key], label='Valid')
            ax[divmod(i, 3)].legend()
        fig.savefig(log_dir / 'metric.svg')
        plt.close()

        print('', flush=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    @torch.no_grad()
    def visulize(engine):
        model.eval()
        epoch_dir = log_dir / f'{engine.state.epoch:03d}'
        epoch_dir.mkdir()
        for i, (inp_b, lbl_b) in enumerate(iter(visul_loader)):
            N, _, H, W = inp_b.size()
            out_b = model(inp_b.to(device)).cpu()
            out_b = out_b.view(N, 1, 1, 1).expand(N, 3, H, W)
            lbl_b = lbl_b.view(N, 1, 1, 1).expand(N, 3, H, W)
            save_image(torch.cat([
                inp_b[:, 0:3],
                util.colorize(inp_b[:, 3:4], plt.cm.gray),
                util.colorize(inp_b[:, 4:5], plt.cm.gray),
                lbl_b,
                out_b,
            ], dim=0), epoch_dir / f'{i:03d}.png', pad_value=1, nrow=N)
        
    trainer.run(train_loader, max_epochs=30)


def infer(ckpt_path):
    txt_path = '../raw/test_public_list.txt'
    test_data = Data('../raw/', txt_path)
    test_loader = DataLoader(test_data, 32, shuffle=False, num_workers=4)

    device = torch.device('cuda')
    model = Model()
    model.load_state_dict(torch.load(ckpt_path))

    pred_dir = Path('./infer/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
    pred_dir.mkdir(parents=True, exist_ok=True)

    inferer = create_supervised_evaluator(model, {}, device)
    ProgressBar(True).attach(inferer)

    result = []
    @inferer.on(Events.ITERATION_COMPLETED)
    def dump_pred(engine):
        pred_b, _ = engine.state.output
        pred_b = pred_b.cpu().numpy().ravel().tolist()
        result.extend(pred_b)

    inferer.run(test_loader, max_epochs=1)

    df = pd.read_csv(txt_path, sep=' ', header=None)
    df[3] = result
    df.to_csv(pred_dir / 'pred.txt', index=None, 
        header=None, sep=' ', float_format='%.7f')

    fig, ax = plt.subplots()
    ax.hist(np.float32(result), bins=50)
    fig.savefig(str(pred_dir / 'hist.svg'))
    plt.close()

if __name__ == '__main__':
    train()
    infer('')
