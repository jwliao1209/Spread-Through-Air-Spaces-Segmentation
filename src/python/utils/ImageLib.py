import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


__all__ = ["ImageLib"]

class ImageLib(object):
    @classmethod
    def ReadAsPIL(cls, imgLoc):
        return Image.open(imgLoc)

    @classmethod
    def ReadAsNumpy(cls, imgLoc):
        return cls.PILToNumpy( cls.ReadAsPIL(imgLoc) )

    @classmethod
    def ReadAsTensor(cls, imgLoc):
        return cls.PILToTensor( cls.ReadAsPIL(imgLoc) )

    @classmethod
    def PILToNumpy(cls, img):
        return np.asarray(img)

    @classmethod
    def PILToTensor(cls, img):
        return TF.to_tensor(img)

    @classmethod
    def NumpyToTensor(cls, img):
        return torch.tensor(img)

    @classmethod
    def NumpyToPIL(cls, img):
        return Image.fromarray(img)

    @classmethod
    def TensorToNumpy(cls, img):
        return img.cpu().detach().permute(1,2,0).numpy()

    @classmethod
    def TensorToPIL(cls, img):
        return T.ToPILImage()(img.cpu().detach())
        
    
