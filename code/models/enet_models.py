import torch.nn as nn
from efficientnet_pytorch import model as enet

class EfficientNetModel(nn.Module):
    def __init__(self, backbone, c_out=5):
        super(EfficientNetModel, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)

        self.myfc = nn.Linear(self.enet._fc.in_features, c_out)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x