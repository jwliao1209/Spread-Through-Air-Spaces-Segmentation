import random
import os.path as osp

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .Transforms import *
from .AutoAug import AutoAugmentation
from ..utils import ReadJson

__all__ = ["stasDataset", "stasCollateFn"]

class stasDataset(Dataset):
    def __init__(self, args, stage='train'):
        self.stage = stage.lower()
        self.checkList = ['train', 'valid', 'self_test', 'public_test', 'private_test']
        assert self.stage in self.checkList
        
        self.jsonRoot = args.BASIC.JSON_ROOT
        self.imgRoot  = args.BASIC.IMG_ROOT

        self.mosaic_prob = args.AUG.MOSAIC.PROB if self.stage=='train' else 0        
        self._setup(args)
        self._set_transforms(args)

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        data = self._basic_transform(idx)
        return data

    def _basic_transform(self, idx):
        data = self._get_dataDict(idx)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def _get_dataDict(self, idx):
        dataPoint = self.dataList[idx]
        fileName = dataPoint['filename']
        if self.stage in ['train', 'valid', 'self_test']:
            data = {
                'ID'   : dataPoint['filename'],
                'image': f"{fileName}.jpg",
                'label': f"{fileName}.png",
                'bbox' : dataPoint['bbox_xyxy']
            }
        else:
            data = {
                'ID' : dataPoint['filename'],
                'image': f"{fileName}.jpg"
            }
        return data

    def _setup(self, args):
        # setup location
        if self.stage == 'train':
            self.labLoc = args.TRAIN.LAB_LOC
            self.imgLoc = args.TRAIN.IMG_LOC
            jsonFile = args.TRAIN.JSON_NAME + '_train.json'
            
        elif self.stage == 'valid':
            self.labLoc = args.TRAIN.LAB_LOC
            self.imgLoc = args.TRAIN.VAL_LOC
            jsonFile = args.TRAIN.JSON_NAME + '_valid.json'
            
        elif self.stage == 'self_test':
            self.labLoc = args.SELF_TEST.LAB_LOC
            self.imgLoc = args.SELF_TEST.IMG_LOC
            jsonFile = 'Test.json'
            
        elif self.stage == 'public_test':
            self.imgLoc = args.PUB_TEST.PUB_IMG_LOC
            jsonFile = 'Public.json'
            
        else: # self.stage == 'private_test'
            self.imgLoc = args.PUB_TEST.PRI_IMG_LOC
            jsonFile = 'Private.json'

        # setup dataList
        self.dataList = ReadJson(osp.join(self.jsonRoot, jsonFile))
        return
        
    def _set_transforms(self, args):
        if self.stage in ['train']:
            self.transform = Compose([
                SetImagePath(keys=['image'], root=self.imgRoot, folder=self.imgLoc, **args.TRAIN),
                SetLabelPath(keys=['label'], root=self.imgRoot, folder=self.labLoc),
                LoadImage(keys=['image', 'label']),
                AutoAugmentation(keys=['image', 'label'], **args.AUG.AUTOAUG),
                ImageToTensor(keys=['image', 'label']),
                RandomBrightness(keys=['image'], **args.AUG.BRIGHT),
                RandomGaussianNoise(keys=['image'], **args.AUG.NOISE),
                Preprocess(keys=['image'], **args.AUG.PREPROCESS),
                # RandomCrop(keys=['image', 'label'], **args.AUG.CROP),
                # Resize(keys=['image', 'label'], **args.AUG.RESIZE)
                RandomFlipLR(keys=['image', 'label'], **args.AUG.FLIPLR),
                RandomFlipUD(keys=['image', 'label'], **args.AUG.FLIPUD),
                # RandomRotate(keys=['image', 'label'], **args.AUG.ROT),
                Pad(keys=['image', 'label'], **args.AUG.PAD)
            ])

            #self.mosaic = Mosaic(keys=['image', 'label'], **args.AUG.MOSAIC)
            
        elif self.stage in ['valid', 'self_test']:
            self.transform = Compose([
                SetImagePath(keys=['image'], root=self.imgRoot, folder=self.imgLoc),
                SetLabelPath(keys=['label'], root=self.imgRoot, folder=self.labLoc),
                LoadImage(keys=['image', 'label']),
                ImageToTensor(keys=['image', 'label']),
                Preprocess(keys=['image'], **args.AUG.PREPROCESS),
                Pad(keys=['image', 'label'], **args.AUG.PAD)
            ])
        else:
            self.transform = Compose([
                SetImagePath(keys=['image'], root=self.imgRoot, folder=self.imgLoc),
                LoadImage(keys=['image']),
                ImageToTensor(keys=['image']),
                Preprocess(keys=['image'], **args.AUG.PREPROCESS)
            ])
        return

def stasCollateFn(data):
    # list of dict -> dict of list
    data = {k: [dic[k] for dic in data] for k in data[0]}
    try:
        data = {k: torch.stack(v, dim=0) if k in ['image', 'label'] else v
                for k, v in data.items()}
    except:
        print(data['ID'])
        print([t.shape for t in data['image']])
        print([t.shape for t in data['label']])
    return data
