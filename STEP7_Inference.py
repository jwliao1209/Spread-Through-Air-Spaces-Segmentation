import os
import glob
import os.path as osp

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Config import get_cfg_defaults
from Checkpoint_setting import ckpt_list
from src.python import (stasDataset, stasCollateFn,
                        ReadYaml, get_activation_layer,
                        get_model, get_model_acc,
                        TTA_Identity, TTA_FlipLR, TTA_FlipUD, TTA_FlipALL,
                        dict_to_cfgNode, dict_to_device,
                        pad_to_match, crop_from_pad)

def get_weight_path(ckptname):
    return osp.join('checkpoint', ckptname, 'model_weights')

def get_config_path(ckptname):
    return osp.join('checkpoint', ckptname, 'config.yaml')

def update_ckpt_list(ckpt_list):
    for ckpt in ckpt_list:
        topk, ckptname = ckpt['topk'], ckpt['ckptname']
        ckpt['pth_root'] = get_weight_path(ckptname)
        ckpt['config'] = dict_to_cfgNode(ReadYaml(get_config_path(ckptname)))
        ckpt['pth_list'] = sorted(glob.glob(osp.join(ckpt['pth_root'], '*.pth')),
                                  key=get_model_acc, reverse=True)[:topk]
    return

def get_model_list(ckpt_list, device):
    model_list = []
    for ckpt in ckpt_list:
        config   = ckpt['config']
        pth_root = ckpt['pth_root']
        pth_list = ckpt['pth_list']
        
        for pth_name in pth_list:
            pth_path = pth_name
            model = get_model(**config).load(pth_path).eval().to(device)
            model_list.append(model)
    return model_list

def get_act_list(ckpt_list):
    act_list = []
    for ckpt in ckpt_list:
        act_name = ckpt['config'].LOSS.ACT_KEY[0]
        topk = ckpt['topk']
        
        act = get_activation_layer(act_name)
        act_list += [act for _ in range(topk)]

    return act_list

def set_up_loader(config, Type='public'):
    assert Type in ['public', 'private']
    
    if Type == 'public':
        img_path = osp.join(config.BASIC.IMG_ROOT, config.PUB_TEST.PUB_IMG_LOC)
    else:
        img_path = osp.join(config.BASIC.IMG_ROOT, config.PUB_TEST.PRI_IMG_LOC)

    if not osp.exists(img_path):
        return None, None, None
    dataset = stasDataset(config, stage=Type+'_test')
    loader = DataLoader(dataset, collate_fn=stasCollateFn)
    saveLoc = osp.join('prediction', Type)
    os.makedirs(saveLoc, exist_ok=True)
    return dataset, loader, saveLoc

def loop_dataset(ckpt_list, model_list, TTA_list, act_list,
                 data_loader, saveLoc, device):
    num_model = len(model_list)
    num_aug = len(TTA_list)
    pad_factor = 32
    cpu = torch.device('cpu')
    for inputDict in tqdm(data_loader):
        # get info
        filename = inputDict['ID'][0]
        _, _, H, W = inputDict['image'].shape
        
        # make augmentation
        Aug = [inputDict.copy() for _ in range(num_aug)]
        Aug = [pad_to_match(dic, ['image'], pad_factor) for dic in Aug]
        Aug = [tta.forward(dic) for tta, dic in zip(TTA_list, Aug)]

        # batch it as a tensor
        inputDict['image'] = torch.cat([dic['image'] for dic in Aug], dim=0)
        inputDict = dict_to_device(inputDict, ['image'], device)
        
        # feed forward
        outDicts = [model.predict(inputDict) for model in model_list]

        # reconstruct the dictionary such that the same augmentation type
        # belongs to the same dict
        Aug = [
            {'stage_0' : torch.cat(
                                    [dic['stage_0'][i].unsqueeze(0)
                                     for dic in outDicts], dim=0
                                    )
            } for i in range(num_aug)
        ]
        Aug = [tta.backward(dic) for tta, dic in zip(TTA_list, Aug)]
        Aug = [dict_to_device(dic, ['stage_0'], cpu) for dic in Aug]

        # crop to original size and vote
        Aug = [crop_from_pad(dic, ['stage_0'], size=(H, W)) for dic in Aug]
        Aug = [dic['stage_0'][:, -1] for dic in Aug]
        Aug = torch.cat(Aug, dim=0)
        num = sum([int(((pred>0.5).int()).sum() > 0) for pred in Aug])
        Pred = torch.sum(Aug, dim=0) / num

        # save image
        save_image(Pred.unsqueeze(0).float(), osp.join(saveLoc, f"{filename}.png"))

if __name__ == '__main__':
    default_cfg = get_cfg_defaults()
    device = torch.device("cuda:0")
    update_ckpt_list(ckpt_list)
    
    # get datasets
    pub_dataset, pub_loader, pub_saveLoc = set_up_loader(default_cfg, Type='public')
    pri_dataset, pri_loader, pri_saveLoc = set_up_loader(default_cfg, Type='private')
    
    # get models
    model_list = get_model_list(ckpt_list, device)

    # get last layer activation
    activation_list = get_act_list(ckpt_list)
    
    # get TTA list
    TTA_list = [
        TTA_Identity(forwardKeys=['image'], backwardKeys=['stage_0']),
        TTA_FlipLR(forwardKeys=['image'], backwardKeys=['stage_0']),
        TTA_FlipUD(forwardKeys=['image'], backwardKeys=['stage_0']),
        TTA_FlipALL(forwardKeys=['image'], backwardKeys=['stage_0'])
    ]
    
    # loop over the datasets
    if pub_dataset is not None:
        print('executing public dataset')
        loop_dataset(ckpt_list, model_list, TTA_list, activation_list,
                     pub_loader, pub_saveLoc, device)

    if pri_dataset is not None:
        print('executing private dataset')
        loop_dataset(ckpt_list, model_list, TTA_list, activation_list,
                     pri_loader, pri_saveLoc, device)

