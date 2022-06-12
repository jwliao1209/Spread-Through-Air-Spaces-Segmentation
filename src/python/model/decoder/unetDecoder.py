import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseModule import (ConvModule, SegmentationHead)
from ..initialize import (initialize_decoder, initialize_head)

__all__ = ['UNetDecoder']

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel,
                 **kwargs):
        assert kwargs.get('NORM') in ['batch', 'layer', 'instance', 'group']
        assert kwargs.get('ACTIVATION') in ['leakyrelu', 'relu', 'silu', 'mish', 'sigmoid']
        assert kwargs.get('UPSCALE') in ['interpolation', 'upconv']

        super().__init__() 
        if kwargs.get('UPSCALE') == 'interpolation':
            self.upsample = lambda x : F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1)

        self.conv1 = ConvModule(
            in_channel=in_channel+skip_channel,
            out_channel=out_channel,
            kernel_size=3,
            padding=1,
            norm_channel=out_channel,
            norm_groups=kwargs.get('NORM_GROUPS'),
            **kwargs)

        self.conv2 = ConvModule(
            in_channel=out_channel,
            out_channel=out_channel,
            kernel_size=3,
            padding=1,
            norm_channel=out_channel,
            norm_groups=kwargs.get('NORM_GROUPS'),
            **kwargs)

        self.conv1x1 = nn.Conv2d(
            in_channels=in_channel+skip_channel,
            out_channels=out_channel,
            kernel_size=1
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        x_skip = self.conv1x1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + x_skip
        

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels,
                 **kwargs):
        '''
        encoder_channels: passed by encoder out_channels
            for example: (3, 32, 24, 40, 112, 320)
        decoder_channels: decoder out_channels
            for example: (16, 32, 64, 128, 256)
        '''
        super().__init__()
        self.n_blocks = len(encoder_channels) - 1
        self.supervision_depth = kwargs.get('DEEP_SUPERVISION_DEPTH')
        
        assert (len(decoder_channels) == self.n_blocks)
        assert (self.supervision_depth >= 1)

        # encoder_channels: (3, 32, 24, 40, 112, 320)->(32, 24, 40, 112, 320)
        encoder_channels = encoder_channels[1:]
        # in_channels = [32, 64, 128, 256, 320]        
        in_channels = list(decoder_channels[1:]) + [encoder_channels[-1]]
        # skip_channels = [3, 32, 24, 40, 112]
        skip_channels = [3] + list(encoder_channels[:-1])
        # out_channels = [16, 32, 64, 128, 256]
        out_channels = decoder_channels

        self.out_channels = decoder_channels
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        segHeads = []
        sd = self.supervision_depth
        for in_chn, act in zip(self.out_channels[:sd], kwargs.get('ACT_KEY')):
            if act == 'softmax':
                segHeads.append(SegmentationHead(in_chn, 2, act))
            else:
                segHeads.append(SegmentationHead(in_chn, 1, act))
        self.segHeads = nn.ModuleList(segHeads)

        initialize_decoder(self.blocks)
        initialize_head(self.segHeads)

    def forward(self, *features):
        '''
        features is a list of tensor
        For default, it should be length 6, the first one is the original image
        '''
        out = {}
        
        # the bottom feature map is the output of the bridge
        # do not join in the skip process
        x = features[-1]
        for stageID in range(self.n_blocks-1, -1, -1):
            skip = features[stageID]
            x = self.blocks[stageID](x, skip)

            if stageID < self.supervision_depth:
                out[_stage(stageID)] = x

        for i, segHead in enumerate(self.segHeads):
            out[_stage(i)] = segHead(out[_stage(i)])

        return out
    
def _stage(idx):
    return f"stage_{idx}"
