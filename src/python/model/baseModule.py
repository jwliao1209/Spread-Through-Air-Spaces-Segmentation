import torch.nn as nn

__all__ = ["ConvModule", "SegmentationHead"]

from .normLayers import get_norm_layer
from .activationLayers import get_activation_layer

class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0,
                 stride=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              stride=stride, padding=padding,
                              bias=False)
        self.norm = get_norm_layer(kwargs.get('NORM'), **kwargs)
        self.act  = get_activation_layer(kwargs.get('ACTIVATION'), **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class SegmentationHead(nn.Module):
    def __init__(self, in_channel, out_channel, activation, kernel_size=3):
        assert activation in [None, 'sigmoid', 'softmax']
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size,
                              padding = kernel_size // 2)

        self.activation = nn.Identity()
        if activation is not None:
            self.activation = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
