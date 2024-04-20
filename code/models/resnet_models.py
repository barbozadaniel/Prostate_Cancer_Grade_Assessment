import torch
import torch.nn as nn
import torchvision.models as models

class AdaptiveConcatPool2d(nn.Module):
    # This layer will concatenate both average and max pool
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class BasicHead(nn.Module):
    # The head of our model
    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.fc = nn.Sequential(AdaptiveConcatPool2d(),
                                Flatten(),
                                nn.Dropout(0.5),
                                nn.Linear(c_in * 2, 512),
                                nn.ReLU(),                  # Need to try other activation functions
                                nn.BatchNorm1d(512),
                                nn.Dropout(0.5),
                                nn.Linear(512, c_out))

    def forward(self, x):
        bn, c, height, width = x.shape
        h = x.view(-1, self.n_tiles, c, height, width).permute(0, 2, 1, 3, 4) \
            .contiguous().view(-1, c, height * self.n_tiles, width)
        h = self.fc(h)
        return h


class ResNetModel(nn.Module):
    # The main model combining a backbone and a head
    def __init__(self, c_out=6, n_tiles=12, tile_size=128, backbone='resnext50_semi', head='basic', **kwargs):
        super().__init__()
        if backbone == 'resnext50_semi':
            m = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        elif backbone == 'resnet50':
            m = models.resnet50(pretrained=True)

        c_feature = list(m.children())[-1].in_features
        self.feature_extractor = nn.Sequential(
            *list(m.children())[:-2])  # Remove resnet head
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        if head == 'basic':
            self.head = BasicHead(c_feature, c_out, n_tiles)

    def forward(self, x):
        h = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(h)
        h = self.head(h)

        return h
