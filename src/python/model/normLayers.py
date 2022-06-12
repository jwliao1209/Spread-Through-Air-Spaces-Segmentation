import torch.nn as nn

__all__ = ["get_norm_layer"]

normLayerCtors = {
    'batch'   : nn.BatchNorm2d,
    'layer'   : nn.GroupNorm,
    'instance': nn.GroupNorm,
    'group'   : nn.GroupNorm
}

def _get_value(name, default=None, **kwargs):
    return default if name not in kwargs else kwargs.get(name)

def _get_input_kwargs(name, **kwargs):
    num_chn = _get_value('norm_channel', default=None, **kwargs)
    num_gps = _get_value('norm_groups', default=8, **kwargs)
    
    if name == 'batch':
        inputDict = {'num_features':num_chn}
    elif name == 'layer':
        inputDict = {'num_groups':num_chn, 'num_channels':num_chn}
    elif name == 'instance':
        inputDict = {'num_groups':1, 'num_channels':num_chn}
    elif name == 'group':
        inputDict = {'num_groups':num_gps, 'num_channels':num_chn}
        
    return inputDict

def get_norm_layer(name, **kwargs):
    name = name.lower()
    assert name in normLayerCtors, f"{name} not in {normLayerCtors.keys()}"
    inputDict = _get_input_kwargs(name, **kwargs)
    return normLayerCtors[name](**inputDict)
