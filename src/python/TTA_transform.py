import torchvision.transforms.functional as F

__all__ = ["TTA_Identity", "TTA_FlipLR", "TTA_FlipUD", "TTA_FlipALL"]

class TTA_Transform(object):
    def __init__(self, forwardKeys, backwardKeys, **kwargs):
        self.f_keys = forwardKeys
        self.b_keys = backwardKeys
        self._parseVariables(**kwargs)

    def __call__(self, data, direction='forward', **kwargs):
        assert direction in ['forward', 'backward']
        if direction == 'forward':
            keys = self.f_keys
            process_fn = self._forward
        else:
            keys = self.b_keys
            process_fn = self._backward

        for key in keys:
            if key in data:
                data[key] = process_fn(data[key], **kwargs)
            else:
                raise KeyError(f"{key} not in data.") 
        return data

    def forward(self, data, **kwargs):
        return self(data, direction='forward', **kwargs)
    
    def backward(self, data, **kwargs):
        return self(data, direction='backward', **kwargs)
    
    def _forward(self, singleInput, **kwargs):
        NotImplementedError
        
    def _backward(self, singleInput, **kwargs):
        NotImplementedError

    def _parseVariables(self, **kwawrgs):
        return

# =============== subclass ===================
class TTA_Identity(TTA_Transform):
    def __init__(self, forwardKeys, backwardKeys, **kwargs):
        super(TTA_Identity, self).__init__(forwardKeys, backwardKeys, **kwargs)

    def _forward(self, singleInput, **kwargs):
        return singleInput

    def _backward(self, singleInput, **kwargs):
        return singleInput
    
class TTA_FlipLR(TTA_Transform):
    def __init__(self, forwardKeys, backwardKeys, **kwargs):
        super(TTA_FlipLR, self).__init__(forwardKeys, backwardKeys, **kwargs)

    def _forward(self, singleInput, **kwargs):
        return F.hflip(singleInput)

    def _backward(self, singleInput, **kwargs):
        return F.hflip(singleInput)

class TTA_FlipUD(TTA_Transform):
    def __init__(self, forwardKeys, backwardKeys, **kwargs):
        super(TTA_FlipUD, self).__init__(forwardKeys, backwardKeys, **kwargs)

    def _forward(self, singleInput, **kwargs):
        return F.vflip(singleInput)

    def _backward(self, singleInput, **kwargs):
        return F.vflip(singleInput)

class TTA_FlipALL(TTA_Transform):
    def __init__(self, forwardKeys, backwardKeys, **kwargs):
        super(TTA_FlipALL, self).__init__(forwardKeys, backwardKeys, **kwargs)

    def _forward(self, singleInput, **kwargs):
        return F.hflip(F.vflip(singleInput))

    def _backward(self, singleInput, **kwargs):
        return F.vflip(F.hflip(singleInput))
