import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .Transforms import (BaseTransform, RandomTransform)

__all__ = ["AutoAugmentation"]

class AutoAugmentation(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(AutoAugmentation, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.opt = kwargs.get('OPT')
        fillcolor = [(128,)*3, (0,)]

        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, data):
        if (random.random() < self.p) and self.opt:
            policy_idx = random.randint(0, len(self.policies)-1)
            data = self.policies[policy_idx](data)
        return data

    def _update_prob(self, cur_ep, total_ep):
        threshold = int((29/30) * total_ep)
        if cur_ep > threshold:
            self.p = 0
        else:
            lamb = cur_ep / threshold
            self.p = lamb * (0.3 * self.init_p) + (1-lamb) * self.init_p
        return

# ========= SubPolicy ===================================
class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1,
                       p2, operation2, magnitude_idx2,
                 fillcolor=[(128,)*3, 0], keys=['image', 'label']):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        ctors = {
            "shearX": ShearX,
            "shearY": ShearY,
            "translateX": TranslateX,
            "translateY": TranslateY,
            "rotate": Rotate,
            "color": Color,
            "posterize": Posterize,
            "solarize": Solarize,
            "contrast": Contrast,
            "sharpness": Sharpness,
            "brightness": Brightness,
            "autocontrast": AutoContrast,
            "equalize": Equalize,
            "invert": Invert
        }
        self.op1 = self._get_tform(p1, operation1, magnitude_idx1, fillcolor, keys, ranges, ctors)                                     
        self.op2 = self._get_tform(p2, operation2, magnitude_idx2, fillcolor, keys, ranges, ctors)

    def _get_tform(self, p, operation, magnitude_idx, fillcolor, keys, ranges, ctors):
        checkList = ["shearX", "shearY", "translateX", "translateY", "rotate"]
        if (operation in checkList) and ('label' in keys):
            op_keys = ['image', 'label']
        else:
            op_keys = ['image']

        return ctors[operation](keys=op_keys, PROB=p, fillcolor=fillcolor,
                                magnitude=ranges[operation][magnitude_idx])

    def __call__(self, data):
        data = self.op1(data)
        data = self.op2(data)
        return data

class ShearX(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(ShearX, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.fillcolor = kwargs.get('fillcolor')
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        rand_num = kwargs.get('rand_num')
        fillcolor = self.fillcolor[0] if singleData.mode == 'RGB' else self.fillcolor[1]
        
        singleData = singleData.transform(
            singleData.size, Image.AFFINE, (1, self.magnitude * rand_num, 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=fillcolor
        )
        return singleData

    def __call__(self, data):
        rand_num = random.choice([-1, 1])
        return super().__call__(data, rand_num=rand_num)

class ShearY(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(ShearY, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.fillcolor = kwargs.get('fillcolor')
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        rand_num = kwargs.get('rand_num')
        fillcolor = self.fillcolor[0] if singleData.mode == 'RGB' else self.fillcolor[1]

        singleData = singleData.transform(
            singleData.size, Image.AFFINE, (1, 0, 0, self.magnitude * rand_num, 1, 0),
            Image.BICUBIC, fillcolor=fillcolor
        )
        return singleData

    def __call__(self, data):
        rand_num = random.choice([-1, 1])
        return super().__call__(data, rand_num=rand_num)

class TranslateX(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(TranslateX, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.fillcolor = kwargs.get('fillcolor')
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        rand_num = kwargs.get('rand_num')
        width = singleData.size[0]
        fillcolor = self.fillcolor[0] if singleData.mode == 'RGB' else self.fillcolor[1]

        singleData = singleData.transform(
            singleData.size, Image.AFFINE, (1, 0, self.magnitude * width * rand_num, 0, 1, 0),
            fillcolor=fillcolor
        )
        return singleData

    def __call__(self, data):
        rand_num = random.choice([-1, 1])
        return super().__call__(data, rand_num=rand_num)

class TranslateY(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(TranslateY, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.fillcolor = kwargs.get('fillcolor')
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        rand_num = kwargs.get('rand_num')
        height = singleData.size[0]
        fillcolor = self.fillcolor[0] if singleData.mode == 'RGB' else self.fillcolor[1]

        singleData = singleData.transform(
            singleData.size, Image.AFFINE, (1, 0, 0, 0, 1, self.magnitude * height * rand_num),
            fillcolor=fillcolor
        )
        return singleData

    def __call__(self, data):
        rand_num = random.choice([-1, 1])
        return super().__call__(data, rand_num=rand_num)


class Rotate(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Rotate, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        rand_num = kwargs.get('rand_num')
        imgMode = singleData.mode
        fillcolor = (128,) * 4 if imgMode == 'RGB' else (0,) * 4
        
        image_rot = singleData.convert("RGBA").rotate(self.magnitude * rand_num)
        background = Image.new("RGBA", image_rot.size, fillcolor)
        singleData = Image.composite(image_rot, background, image_rot).convert(imgMode)
        return singleData

    def __call__(self, data):
        rand_num = random.choice([-1, 1])
        return super().__call__(data, rand_num=rand_num)

class Color(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Color, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        return ImageEnhance.Color(singleData).enhance(1 + self.magnitude * random.choice([-1, 1]))

class Contrast(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Contrast, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        return ImageEnhance.Contrast(singleData).enhance(1 + self.magnitude * random.choice([-1, 1]))

class Sharpness(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Sharpness, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        return ImageEnhance.Sharpness(singleData).enhance(1 + self.magnitude * random.choice([-1, 1]))

class Brightness(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Brightness, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        return ImageEnhance.Brightness(singleData).enhance(1 + self.magnitude * random.choice([-1, 1]))

class Posterize(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Posterize, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        return ImageOps.posterize(singleData, self.magnitude)

class Solarize(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Solarize, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)
        self.magnitude = kwargs.get('magnitude')

    def _process(self, singleData, **kwargs):
        return ImageOps.solarize(singleData, self.magnitude)

class AutoContrast(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(AutoContrast, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)

    def _process(self, singleData, **kwargs):
        return ImageOps.autocontrast(singleData)

class Equalize(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Equalize, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)

    def _process(self, singleData, **kwargs):
        return ImageOps.equalize(singleData)

class Invert(RandomTransform):
    def __init__(self, keys, **kwargs):
        super(Invert, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        super()._parseVariables(**kwargs)

    def _process(self, singleData, **kwargs):
        return ImageOps.invert(singleData)
