import torch
import torch.nn as nn

from .encoder import EfficientNetEncoder
from .decoder import UNetDecoder

__all__ = ["EfficientUNet"]

class EfficientUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        encoder_kwargs = kwargs.get('MODEL').get('ENCODER')
        decoder_kwargs = kwargs.get('MODEL').get('DECODER')
        decoder_actkey = kwargs.get('LOSS').get('ACT_KEY')

        self.depth = kwargs.get('MODEL').get('ENCODER').get('DEPTH')
        self.encoder = EfficientNetEncoder(**encoder_kwargs)
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=[16, 32, 64, 128, 256],
            ACT_KEY=decoder_actkey,
            **decoder_kwargs
        )

    def data_parallel(self, device_ids):
        self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids)
        self.decoder = nn.DataParallel(self.decoder , device_ids=device_ids)

        return

    def forward(self, inputDict):
        features = self.encoder(inputDict)
        outDict = self.decoder(*features)
        return outDict

    @torch.no_grad()
    def predict(self, inputDict):
        return self.forward(inputDict)

    def save(self, path):
        torch.save({
            'Encoder': self.encoder.module.state_dict(),
            'Decoder': self.decoder.module.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(ckpt['Encoder'])
        self.decoder.load_state_dict(ckpt['Decoder'])

        return self
