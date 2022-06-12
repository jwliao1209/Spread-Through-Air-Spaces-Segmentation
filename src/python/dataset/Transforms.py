import os.path as osp
import random
import math

import torch
import torchvision.transforms.functional as F

from ..utils import ImageLib

__all__ = ["LoadImage", "ImageToTensor", "RandomCrop", "Resize",
           "RandomBrightness", "RandomGaussianNoise", "Mosaic",
           "RandomFlipLR", "RandomFlipUD", "RandomRotate",
           "Preprocess", "Pad", "SetImagePath", "SetLabelPath"]

# ======== inherited by other class ==============
class BaseTransform(object):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parseVariables(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data")
        return data

    def _parseVariables(self, **kwargs):
        pass

    def _process(self, singleData, **kwargs):
        NotImplementedError

    def _update_prob(self, cur_ep, total_ep):
        pass

class RandomTransform(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(RandomTransform, self).__init__(keys, **kwargs)

    def __call__(self, data, **kwargs):
        if random.uniform(0, 1) < self.p:
            return super().__call__(data, **kwargs)
        return data

    def _parseVariables(self, **kwargs):
        self.p = kwargs.get('PROB')
        self.init_p = self.p

    def _update_prob(self, cur_ep, total_ep):
        threshold = int(0.9 * total_ep)
        if cur_ep > threshold:
            self.p = 0
        else:
            lamb = cur_ep / threshold
            self.p = lamb * (0.1 * self.init_p) + (1-lamb) * self.init_p
        return

    def _print_info(self):
        print(type(self), self.p)

# ========== FileIO and tensor format transforms ================
class SetLabelPath(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(SetLabelPath, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        self.roots = kwargs.get('root')
        self.folder = kwargs.get('folder')

    def _process(self, singleData, **kwargs):
        return osp.join(self.roots, self.folder, singleData)

class SetImagePath(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(SetImagePath, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        self.root = kwargs.get('root')
        self.folder = kwargs.get('folder')
        
        self.prob = kwargs.get('IMG_PROB') if isinstance(self.folder, list) else None
        self.init_p = self.prob
        self.main_id = None if self.prob is None else self.prob.index(max(self.prob))
        
    def _process(self, singleData, **kwargs):
        if self.prob is None:
            return osp.join(self.root, self.folder, singleData)
        else:
            folder = random.choices(self.folder, self.prob, k=1)[0]
            return osp.join(self.root, folder, singleData)

    def _update_prob(self, cur_ep, total_ep):
        if self.prob is None:
            return
        
        threshold = int(0.9 * total_ep)
        for i, prob in enumerate(self.init_p):
            if i == self.main_id:
                continue
            if cur_ep > threshold:
                self.prob[i] = 0
            else:
                lamb = cur_ep / threshold
                self.prob[i] = lamb * (0.3 * prob) + (1-lamb) * prob
        return
    
# read image as PIL image (filename -> PIL image)
class LoadImage(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(LoadImage, self).__init__(keys, **kwargs)

    def _process(self, singleData, **kwargs):
        return ImageLib.ReadAsPIL(singleData)

# convert PIL image to tensor (PIL image -> torch.Tensor)
class ImageToTensor(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ImageToTensor, self).__init__(keys, **kwargs)
        
    def _process(self, singleData, **kwargs):
        return ImageLib.PILToTensor(singleData)

# ================== tensor value transform ========================
class Preprocess(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(Preprocess, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        self.mean = torch.tensor(kwargs.get('MEAN')).view(3,1,1)
        self.std = torch.tensor(kwargs.get('STD')).view(3,1,1)

    def _process(self, singleData, **kwargs):
        return (singleData - self.mean) / self.std

class RandomBrightness(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(RandomBrightness, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.factor = kwargs.get('FACTOR')

    def _process(self, singleData, **kwargs):
        singleData = F.adjust_brightness(singleData, kwargs.get('factor'))
        singleData = torch.clip(singleData, min=0, max=1)
        return singleData

    def __call__(self, data):
        factor = random.uniform(1-self.factor, 1+self.factor)
        return super().__call__(data, factor=factor)
            
class RandomGaussianNoise(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(RandomGaussianNoise, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.sig = kwargs.get('SIGMA')

    def _process(self, singleData, **kwargs):
        singleData += self.sig * torch.randn(singleData.shape)
        return singleData

# =================== Rigid transform ===================
class RandomFlipLR(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(RandomFlipLR, self).__init__(keys, **kwargs)

    def _process(self, singleData, **kwargs):
        singleData = F.hflip(singleData)
        return singleData

class RandomFlipUD(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(RandomFlipUD, self).__init__(keys, **kwargs)

    def _process(self, singleData, **kwargs):
        singleData = F.vflip(singleData)
        return singleData

# Rotate 90/180/270 degrees randomly
class RandomRotate(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(RandomRotate, self).__init__(keys, **kwargs)

    def _process(self, singleData, **kwargs):
        k = random.randint(1,3) # 90, 180, 270 degrees
        singleData = torch.rot90(singleData, k=k, dims=[1,2])
        return singleData

# Random crop an sub-image from the original image (tensor of size C,H,W)
class RandomCrop(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(RandomCrop, self).__init__(keys, **kwargs)
        
    def _parseVariables(self, **kwargs):
        self.randP = kwargs.get('PROB_STAS')
        self.cropRange = kwargs.get('RAND_RANGE')

    def _process(self, singleData, **kwargs):
        top, bottom = kwargs.get('top'), kwargs.get('bottom')
        left, right = kwargs.get('left'), kwargs.get('right')
        return singleData[:, top:bottom, left:right]

    def _randomRange(self, imgS, cropS, boxBegin, boxEnd):
        # predefine variables
        imgBegin, imgEnd = 0, imgS-1
        # convert bounding box from float to int
        boxBegin = max(math.floor(boxBegin), imgBegin)
        boxEnd   = min(math.ceil(boxEnd), imgEnd)
        boxS = boxEnd-boxBegin+1

        if boxS < cropS:
            # if crop size is greater than box size
            # then there exist a bounding box contains whole bounding box
            # just find it
            lowBound = max(0, boxEnd-cropS)
            uppBound = min(boxBegin, imgEnd-cropS)
            cropBegin = random.randint(lowBound, uppBound)
            cropEnd = cropBegin + cropS
        else:
            # If there doesn't exist such a bounding box
            # Then try to preserve some air in the boundary

            # preserve air in the Left/Up
            try:
                leftLowBound = math.floor(0.25*boxBegin)
                leftUppBound = min(math.ceil(0.75*boxBegin), imgEnd-cropS)
                cropBegin = random.randint(leftLowBound, leftUppBound)
            except:
                cropBegin = None

            # preserve air in the Right/Bottom
            try:
                rightLowBound = max(math.floor(0.25*boxEnd+0.75*imgEnd), cropS)
                rightUppBound = math.ceil(0.75*boxEnd+0.25*imgEnd)
                cropEnd = random.randint(rightLowBound, rightUppBound)
            except:
                cropEnd = None

            if (cropBegin is not None) and (cropEnd is not None):
                if random.choice(['left', 'right']) == 'left':
                    cropEnd = cropBegin + cropS
                else:
                    cropBegin = cropEnd - cropS
            elif cropBegin is not None:
                cropEnd = cropBegin + cropS
            else:
                cropBegin = cropEnd - cropS
                
        return cropBegin, cropEnd
        
    def __call__(self, data):
        '''
        Determine the left top coordinate
        '''
        # Predefine variables
        imgH, imgW   = data['image'].shape[1:]
        outSize = random.randint(*self.cropRange)
        cropH, cropW = outSize, outSize

        if random.uniform(0, 1) <= self.randP:
            x1, y1, x2, y2 = random.choice(data['bbox'])
            top, bottom = self._randomRange(imgH, cropH, y1, y2)
            left, right = self._randomRange(imgW, cropW, x1, x2)
        else:
            top    = random.choice(range(0, imgH-cropH-1))
            left   = random.choice(range(0, imgW-cropW-1))
            bottom = top + cropH
            right  = left + cropW

        return super().__call__(data,
                                top=top, left=left, bottom=bottom, right=right)

# ================== distortion transform ====================
class Resize(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(Resize, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        self.size = kwargs.get('SIZE')

    def _process(self, singleData, **kwargs):
        return F.resize(singleData, size=self.size, interpolation=F.InterpolationMode.NEAREST)

class Pad(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(Pad, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        self.size = kwargs.get('SIZE')

    def _process(self, singleData, **kwargs):
        L = kwargs.get('leftpad')
        R = kwargs.get('rightpad')
        T = kwargs.get('toppad')
        B = kwargs.get('lowpad')
        return F.pad(singleData, (L, T, R, B), padding_mode='reflect')

    def __call__(self, data, **kwargs):
        _, iH, iW = data['image'].shape
        pH, pW = self.size
        rH, rW = pH - iH, pW - iW
        hH, hW = rH // 2, rW // 2
        return super().__call__(data,
                                leftpad=hW, rightpad=rW-hW,
                                toppad=hH, lowpad=rH-hH)

class Mosaic(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Mosaic, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        self.inSize  = kwargs.get('IN_SIZE')
        self.outSize = kwargs.get('OUT_SIZE')

    def _process(self, singleData, **kwargs):
        top, bottom = kwargs.get('top'), kwargs.get('bottom')
        left, right = kwargs.get('left'), kwargs.get('right')
        
        upperHalf = torch.cat(singleData[:2], dim=2)
        lowerHalf = torch.cat(singleData[2:], dim=2)
        full = torch.cat([upperHalf, lowerHalf], dim=1)      
        return full[:, top:bottom, left:right]        
        
    def __call__(self, data, **kwargs):
        inH, inW   = self.inSize
        outH, outW = self.outSize
        centerH, centerW = inH - 1, inW - 1
        
        quadH, quadW = int(inH / 4), int(inW / 4)
        dh = random.randint(-3*quadH, -quadH)
        dw = random.randint(-3*quadW, -quadW)
        top, left = int(centerH+dh), int(centerW+dw)
        bottom, right = top + outH, left + outW
        return super().__call__(data, top=top, left=left, bottom=bottom, right=right)
