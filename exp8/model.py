import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34

class CASIASURFModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(pretrained=False)
        self.features = nn.Sequential(
            nn.Conv2d(5, 64, (7, 7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, inp_b):
        N = inp_b.size(0)
        fmp_b = self.features(inp_b)
        vec_b = F.adaptive_avg_pool2d(fmp_b, (1, 1))
        vec_b = vec_b.view(N, -1)
        out_b = self.regressor(vec_b)
        return out_b
        

if __name__ == '__main__':
    device = torch.device('cuda')
    model = CASIASURFModel().to(device)
    N, H, W = 32, 64, 64
    inp_b = torch.rand(N, 5, H, W).to(device)
    out_b = model(inp_b)
    print(out_b.size())
    input()
