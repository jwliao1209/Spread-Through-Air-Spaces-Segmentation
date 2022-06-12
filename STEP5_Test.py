import os
import os.path as osp
import argparse

import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Config import get_ckpt_cfg
from src.python import (stasDataset, stasCollateFn,
                        dict_to_device, dict_to_cfgNode,
                        pad_to_match, crop_from_pad,
                        get_model, get_model_acc, img_to_binary, compute_acc,
                        Logger, AvgMeter, ReadYaml)

def prepare(config):
    device = torch.device(f"cuda:{config.BASIC.DEVICE[0]}")
    model = get_model(**config)
    TestDataset = stasDataset(config, stage='self_test')
    TestLoader = DataLoader(dataset=TestDataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=stasCollateFn)

    return device, model, TestLoader


def init_meter():
    meters = {
        'dice'      : AvgMeter(),
        'recall'    : AvgMeter(),
        'precision' : AvgMeter()
    }

    return meters


def cat_images(*images):
    images = filter(lambda x : x is not None, images)
    images = [img.squeeze(0) if img.dim()==4 else img for img in images]
    images = [img.expand(3,-1,-1) if img.size(0)==1 else img for img in images]

    return torch.cat(images, dim=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='', help='XXXX-XX-XX-XX-XX-XX')
    args = parser.parse_args()

    ckpt_root   = osp.join('checkpoint', args.checkpoint)
    weight_root = osp.join(ckpt_root, 'model_weights')
    save_root   = osp.join(ckpt_root, 'self_test')
    imgPred_root = osp.join(save_root, 'image_pred')
    csvPred_root = osp.join(save_root, 'csv_pred')
    
    weight_list = os.listdir(weight_root)
    weight_list = sorted(weight_list, key=get_model_acc, reverse=True)
    os.makedirs(imgPred_root, exist_ok=True)
    os.makedirs(csvPred_root, exist_ok=True)
    
    config = dict_to_cfgNode(ReadYaml(osp.join(ckpt_root, 'config.yaml')))
    device, model, TestLoader = prepare(config)
    logger = Logger(print_=False)
    meters = init_meter()
    save_pred = True

    model.to(device).eval()
    for weight_name in weight_list:
        logger.reset()
        [meters[k].reset() for k in meters]
        
        tqdm_bar = tqdm(TestLoader)
        model.load(osp.join(weight_root, weight_name))
        
        for batch_input in tqdm_bar:
            _, _, H, W = batch_input['image'].shape
            
            batch_input = pad_to_match(batch_input, keys=['image'], factor=(2**config.MODEL.ENCODER.DEPTH))
            batch_input = dict_to_device(batch_input, keys=['image', 'label'], device=device)
            
            batch_output = model.predict(batch_input)
            batch_output = crop_from_pad(batch_output, keys=['stage_0'], size=(H, W))
            
            dice, recall, precision = compute_acc(batch_output, batch_input)
            dice, recall, precision = dice.item(), recall.item(), precision.item()
            
            meters['dice'].update(dice, 1)
            meters['recall'].update(recall, 1)
            meters['precision'].update(precision, 1)

            tqdm_bar.set_postfix({k:v.item() for k,v in meters.items()})
            logger.add(
                ID=batch_input['ID'][0],
                dice=dice,
                recall=recall,
                precision=precision
            )

            if save_pred:
                batch_input = crop_from_pad(batch_input, keys=['image'], size=(H, W))
                img = batch_input['image']
                lab = batch_input['label']
                pred = batch_output['stage_0']
                pred = img_to_binary(pred, img_type='softmax' if (pred.size(1) > 1) else 'sigmoid')

                cat = cat_images(lab, img, pred)
                save_path = osp.join(imgPred_root, batch_input['ID'][0]+'.jpg')
                save_image(cat, save_path)
        
        save_pred &= False
        logger.add(
            ID=None,
            dice=meters['dice'].item(),
                recall=meters['recall'].item(),
                precision=meters['precision'].item()
        )
        logger.save(osp.join(csvPred_root, weight_name.replace('.pth', '.csv')))
