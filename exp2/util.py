import torch
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, confusion_matrix


class RunningMean:
    def __init__(self):
        self.iter = 0
        self.sum = 0.0

    def update(self, x):
        self.sum += x
        self.iter += 1

    def value(self):
        if self.iter == 0:
            return float('nan')
        return self.sum / self.iter

    def __str__(self, fmt='{:.5f}'):
        return fmt.format(self.value())


class RunningData:
    def __init__(self):
        self.data = []
    
    def update(self, x):
        self.data.append(x)

    def value(self):
        return torch.cat(self.data, dim=0)


def roc(ax, true, pred):
    true = true.ravel()
    pred = pred.ravel()

    fpr, tpr, thr = roc_curve(true, pred)
    f_tpr = interp1d(fpr, tpr)

    ax.plot(fpr, tpr, '-')
    ax.set_xscale('log')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_xlim(left=5e-5, right=5e0)
    ax.grid(True, which='both')

    pred = (pred > 0.5)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    acc = (tp + tn) / pred.shape[0]
    apcer = fp / (tn + fp)
    npcer = fn / (fn + tp)
    acer = (apcer + npcer) / 2

    return {
        'auc': auc(fpr, tpr),
        'tpr@1e-2': f_tpr(np.array([1e-2]))[0],
        'tpr@1e-3': f_tpr(np.array([1e-3]))[0],
        'tpr@1e-4': f_tpr(np.array([1e-4]))[0],
        'acc': acc, 
        'acer': acer,
        'apcer': apcer,
        'npcer': npcer,
    }