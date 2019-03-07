import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn import metrics as skmetric

from ignite.metrics import Metric

def roc_metric(y_true, y_pred):
    fpr, tpr, thresh = skmetric.roc_curve(y_true, y_pred)
    f = interp1d(fpr, tpr)

    return {
        'auc': skmetric.auc(fpr, tpr),
        'tpr@fpr=1e-4': f(np.float32([1e-4]))[0],
        'tpr@fpr=1e-3': f(np.float32([1e-3]))[0],
        'tpr@fpr=1e-2': f(np.float32([1e-2]))[0],
    }

def cls_metric(y_true, y_pred, thresh=0.5):
    y_true = (y_true > thresh).astype(np.int32)
    y_pred = (y_pred > thresh).astype(np.int32)
    tn, fp, fn, tp = skmetric.confusion_matrix(y_true, y_pred).ravel()

    return {
        'acc': (tn + tp) / (tn + fp + fn + tp + 1e-8),
        'fp': fp,
        'fn': fn,
        'acer': 0.5 * (fp / (tn + fp + 1e-8) + fn / (fn + tp + 1e-8))
    }


class MyMetric(Metric):
    def __init__(self, criterion, output_transform=lambda x: x):
        self.criterion = criterion
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.preds = []
        self.trues = []

    def update(self, output):
        y_pred, y_true = output
        self.preds.append(y_pred.detach().cpu())
        self.trues.append(y_true.detach().cpu())

    def compute(self):
        y_pred = torch.cat(self.preds, dim=0)
        y_true = torch.cat(self.trues, dim=0)
        loss = self.criterion(y_pred, y_true).item()
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        m1 = roc_metric(y_true, y_pred)
        m2 = cls_metric(y_true, y_pred)
        return {'loss': loss, **m1, **m2}



def kpt2bbox(kpt, r=[2, 2]):
    r = torch.FloatTensor(r)
    xy_min, _ = kpt.min(dim=0)
    xy_max, _ = kpt.max(dim=0)
    wh = xy_max - xy_min
    xy_ctr = 0.5 * (xy_min + xy_max)
    xy_min = xy_ctr - 0.5 * (wh * r)
    xy_max = xy_ctr + 0.5 * (wh * r)
    bbox = torch.cat([xy_min, xy_max], dim=0) # xyxy
    bbox = bbox[[1, 0, 3, 2]] # rcrc
    return bbox


def bbox_fit(bbox, size):
    w, h = size
    bbox = (bbox * torch.FloatTensor([h, w, h, w])).long()
    bbox[[0, 2]] = torch.clamp(bbox[[0, 2]], min=0, max=h-1) # r
    bbox[[1, 3]] = torch.clamp(bbox[[1, 3]], min=0, max=w-1) # c
    return bbox


def colorize(tensor, colormap=plt.cm.jet):
    '''Apply colormap to tensor
    Args:
        tensor: (FloatTensor), sized [N, 1, H, W]
        colormap: (plt.cm.*)
    Return:
        tensor: (FloatTensor), sized [N, 3, H, W]
    '''
    tensor = tensor.clamp(min=0.0)
    tensor = tensor.squeeze(dim=1).numpy() # [N, H, W]
    tensor = colormap(tensor)[..., :3] # [N, H, W, 3]
    tensor = torch.from_numpy(tensor).float()
    tensor = tensor.permute(0, 3, 1, 2) # [N, 3, H, W]
    return tensor

def normalize(tensor, eps=1e-8):
    '''Normalize each tensor in mini-batch like Min-Max Scaler
    Args:
        tensor: (FloatTensor), sized [N, C, H, W]
    Return:
        tensor: (FloatTensor) ranged [0, 1], sized [N, C, H, W]
    '''
    N = tensor.size(0)
    min_val = tensor.contiguous().view(N, -1).min(dim=1)[0]
    tensor = tensor - min_val.view(N, 1, 1, 1)
    max_val = tensor.contiguous().view(N, -1).max(dim=1)[0]
    tensor = tensor / (max_val + eps).view(N, 1, 1, 1)
    return tensor


class GradCam:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.fmap_pool = dict()
        self.grad_pool = dict()

        def forward_hook(module, input, output):
            self.fmap_pool[module] = output.detach().cpu()
        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[module] = grad_out[0].detach().cpu()
        
        for layer in layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        assert layer in self.layers, f'{layer} not in {self.layers}'
        fmap_b = self.fmap_pool[layer] # [N, C, fmpH, fmpW]
        grad_b = self.grad_pool[layer] # [N, C, fmpH, fmpW]

        grad_b = F.adaptive_avg_pool2d(grad_b, (1, 1)) # [N, C, 1, 1]
        gcam_b = (fmap_b * grad_b).sum(dim=1, keepdim=True) # [N, 1, fmpH, fmpW]
        gcam_b = F.relu(gcam_b)

        return gcam_b


class GuidedBackPropogation:
    def __init__(self, model):
        self.model = model
        self.hooks = []

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return tuple(grad.clamp(min=0.0) for grad in grad_in)

        for name, module in self.model.named_modules():
            self.hooks.append(module.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)
    
    def get(self, layer):
        return layer.grad.cpu()

