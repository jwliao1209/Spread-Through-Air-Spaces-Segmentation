import torch.nn as nn

__all__ = ["get_activation_layer"]

actLayerCtor = {
    'leakyrelu': nn.LeakyReLU,
    'relu'     : nn.ReLU,
    'silu'     : nn.SiLU,
    'sigmoid'  : nn.Sigmoid,
    'mish'     : nn.Mish,
    'softmax'  : nn.Softmax2d
}

def _get_value(name, default=None, **kwargs):
    return default if name not in kwargs else kwargs.get(name)

def _get_input_kwargs(name, **kwargs):
    inplace = _get_value('inplace', default=True, **kwargs)
    neg_slope = _get_value('negative_slope', default=0.01, **kwargs) # for leakyrelu

    if name == 'leakyrelu':
        inputDict = {'negative_slope':neg_slope, 'inplace':inplace}
    elif name == 'relu':
        inputDict = {'inplace':inplace}
    elif name == 'silu':
        inputDict = {'inplace':inplace}
    elif name == 'sigmoid':
        inputDict = {}
    elif name == 'mish':
        inputDict = {'inplace':inplace}
    elif name == 'softmax':
        inputDict = {}
    return inputDict

def get_activation_layer(name, **kwargs):
    name = name.lower()
    assert name in actLayerCtor, f"{name} is not in {actLayerCtor.keys()}"
    inputDict = _get_input_kwargs(name, **kwargs)
    return actLayerCtor[name](**inputDict)
