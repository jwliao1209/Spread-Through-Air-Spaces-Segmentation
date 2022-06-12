import timm
import torch
import torch.nn as nn

__all__ = ["EfficientNetEncoder"]

'''
EfficientNetEncoder is an encoder + bridge structure in UNet architecture.
We use Efficient net as the backbone, so that we may get a better result.

The encoder+bridge structure looks like
[
    nn.Identity(),
    Efficient net layers downsampling by a factor 2
    ...
    Efficient net layers downsampling by a factor 2 <- treated as bridge
]

Since efficient has only downsample by a factor 32 (i.e, 5 times)
Therefore, the depth of user setting is at most 5.
'''

class EfficientNetEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        assert (kwargs.get('DEPTH') in range(1,6)), "depth should between 1 to 5"
        assert (kwargs.get('NAME').startswith('efficientnet_b'))
        
        self.name = kwargs.get('NAME')
        self.depth = kwargs.get('DEPTH')
        self.drop_rate = kwargs.get('DROP_RATE')
        self.timm_indices = timm_efficientnet_settings[self.name]['stage_idxs']
        self.out_channels = timm_efficientnet_settings[self.name]['out_channels']
        
        self._make_layers()

    def forward(self, inputDict):
        x, Features = inputDict['image'], []
        for stage in self.stages:
            x = stage(x)
            Features.append(x)
        return Features
    
    def _make_layers(self):
        depth = self.depth
        drop_rate = self.drop_rate
        timm_idxs = self.timm_indices
        model_name = self.name

        Enet = timm.create_model(
            model_name=model_name,
            drop_rate=drop_rate,
            pretrained=True,
            features_only=True
        )
        Enet_childs = list(Enet.children())

        First_stage = Enet_childs[:3]
        Other_stage = list(Enet_childs[-1].children())

        stages = []
        stages.append(nn.Identity())
        stages.append(nn.Sequential(*First_stage))
        for stage in range(2, depth+1):
            if stage == 2:
                begin = 0
                end   = timm_idxs[0]
            elif stage == 5:
                begin = timm_idxs[-1]
                end   = len(Other_stage)
            else:
                begin = timm_idxs[stage-3]
                end   = timm_idxs[stage-2]
            stages.append(nn.Sequential(*Other_stage[begin:end]))
            
        self.stages = nn.ModuleList(stages)
        return
        

# settings for timm-efficientnets
timm_efficientnet_settings = {
    "efficientnet_b0": {
        "out_channels": (3, 32, 24, 40, 112, 320),
        "stage_idxs": (2, 3, 5)
    },
    "efficientnet_b1": {
        "out_channels": (3, 32, 24, 40, 112, 320),
        "stage_idxs": (2, 3, 5)
    },
    "efficientnet_b2": {
        "out_channels": (3, 32, 24, 48, 120, 352),
        "stage_idxs": (2, 3, 5)
    },
    "efficientnet_b3": {
        "out_channels": (3, 40, 32, 48, 136, 384),
        "stage_idxs": (2, 3, 5)
    },
    "efficientnet_b4": {
        "out_channels": (3, 48, 32, 56, 160, 448),
        "stage_idxs": (2, 3, 5)
    }
}
