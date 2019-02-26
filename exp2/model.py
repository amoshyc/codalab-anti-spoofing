import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet121


class CASIASURFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = densenet121().features
        self.features.conv0 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.regressor = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x


if __name__ == '__main__':
    device = 'cuda'
    model = CASIASURFModel().to(device)
    img_b = torch.rand(16, 5, 64, 64).to(device)
    lbl_b = torch.rand(16, 1).to(device)
    out_b = model(img_b)
    print(out_b.size())

    loss = nn.BCELoss()(out_b, lbl_b)
    loss.backward()