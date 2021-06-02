import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

# 12 layers!
# k: number of filters to use in intermediate conv layers
# parallels AlphaGo + res layers
class convNet(nn.Module):
    def __init__(self, k, batchNorm = False):
        super(convNet, self).__init__()
        if batchNorm:
            self.model = nn.Sequential(
                layers.symConv2D(55, k, 5),
                nn.BatchNorm2d(k),
                nn.ReLU(),
                layers.resBlock(k),
                layers.resBlock(k),
                layers.resBlock(k),
                layers.resBlock(k),
                layers.resBlock(k), # resBlock # 5 (layer 11)
                layers.OutputLayer(k) # layer 12
            )
        else:
            self.model = nn.Sequential(
                layers.symConv2D(55, k, 5),
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.symConv2D(k, k, 3), # layer 5
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.symConv2D(k, k, 3), # layer 10
                nn.ReLU(),
                layers.symConv2D(k, k, 3),
                nn.ReLU(),
                layers.OutputLayer(k) # layer 12
            )

    def forward(self, x):
        return self.model(x)