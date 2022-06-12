import os
import glob
import random

import torch
import numpy as np
import torch.nn.functional as F
from yacs.config import CfgNode as CN

__all__ = ["get_model_acc", "save_model", "set_determinism",
           "dict_to_device", "cfgNode_to_dict", "dict_to_cfgNode",
           "pad_to_match", "crop_from_pad"]
           
# ================= pth file related ===============
def get_model_acc(modelPath):
    return float(modelPath.split(os.sep)[-1][-10:-4])

def save_model(root, model, epoch, acc, weight_num):
    filename = f"epoch={str(epoch):0>4}-acc={acc:.5f}.pth"
    save_path = os.path.join(root, filename)
    model.save(save_path)
    weight_list = sorted(glob.glob(os.path.join(root, '*.pth')), key=get_model_acc, reverse=True)

    if len(weight_list) > weight_num:
        os.remove(weight_list[-2])

    return

# =============== device related ==================
def dict_to_device(dataDict, keys, device):
    for key in keys:
        if key in dataDict:
            dataDict[key] = dataDict[key].to(device)
        else:
            raise KeyError(f"key should in {keys}, not {key}")
    return dataDict

def set_determinism(seed, benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark
    return

# ============== config related ===================
def cfgNode_to_dict(cfgNode):
    if not isinstance(cfgNode, dict):
        return cfgNode

    for key, val in cfgNode.items():
        cfgNode[key] = cfgNode_to_dict(val)
    
    return dict(cfgNode.items())

def dict_to_cfgNode(Dict):
    _C = CN()
    for k, v in Dict.items():
        if isinstance(v, float):
            _C[k] = int(v) if v.is_integer() else v
        elif not isinstance(v, dict):
            _C[k] = v
        else:
            _C[k] = dict_to_cfgNode(v)
    return _C

# =============== image related ==============
def pad_to_match(inputDict, keys, factor):
    assert 'image' in inputDict
    _, _, H, W = inputDict['image'].shape
    
    # how many pixed will be padded
    pH, pW = factor - H % factor, factor - W % factor
    # half pad size
    hH, hW = pH // 2, pW // 2

    for key in keys:
        img = inputDict[key]
        inputDict[key] = F.pad(img, (hW, pW-hW, hH, pH-hH))
    return inputDict

def crop_from_pad(inputDict, keys, size=(942, 1716)):
    assert ('stage_0' in inputDict) or ('image' in inputDict)
    oH, oW = size
    try:
        _, _, nH, nW = inputDict['stage_0'].shape
    except:
        _, _, nH, nW = inputDict['image'].shape

    # get the start index
    hH, hW = (nH - oH) // 2, (nW - oW) // 2

    for key in keys:
        img = inputDict[key]
        inputDict[key] = img[:,:,hH:hH+oH, hW:hW+oW]
    return inputDict
