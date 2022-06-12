import os
import os.path as osp
from datetime import datetime

import torch
import torchvision.transforms.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


from .Settings import (get_model, get_optimizer, get_scheduler, get_criterion)
from .dataset import (stasDataset, stasCollateFn)
from .Metrics import compute_acc
from .utils import (AvgMeter, Logger, WriteYaml, save_model, dict_to_device,
                    pad_to_match, crop_from_pad)

__all__ = ["trainer"]

def init_meter(Type):
    meter = {
        'loss': AvgMeter(),
        'dice': AvgMeter()
    }
    if Type == 'valid': return meter
    meter['lr'] = AvgMeter()

    return meter

def update_loader(dataloader, cur_ep, total_ep):
    for tform in dataloader.dataset.transform.transforms:
        tform._update_prob(cur_ep, total_ep)
    return

def label_downsample(inputDict, down_times):
    N, C, H, W = inputDict['label'].shape
    for i in range(down_times):
        if i == 0:
            inputDict[f"label_{i}"] = inputDict['label']
        else:
            factor = 2**i
            size = (int(H/factor), int(W/factor))
            inputDict[f"label_{i}"] = F.resize(
                inputDict['label'], size=size,
                interpolation=F.InterpolationMode.NEAREST)

    return inputDict


def training_step(ep, model, train_loader, criterion, optimizer, scheduler,
                  device, **kwargs):
    model.train()
    optimizer.zero_grad()
    meter = init_meter('train')
    tqdm_bar = tqdm(train_loader, desc=f'Training {ep}')
    for i, batch_inputs in enumerate(tqdm_bar):
        batch_size = batch_inputs['image'].size(0)
        # batch_inputs = pad_to_match(batch_inputs, ['image', 'label'], 2**model.depth)
        batch_inputs = label_downsample(batch_inputs, kwargs.get('down_times'))
        batch_inputs = dict_to_device(batch_inputs,
                                      keys=kwargs.get('device_keys'),
                                      device=device)
        batch_outputs = model(batch_inputs)
        batch_inputs  = crop_from_pad(batch_inputs, ['image', 'label', 'label_0'])
        batch_outputs = crop_from_pad(batch_outputs, ['stage_0'])

        losses = criterion(batch_outputs, batch_inputs)
        losses.backward()

        if (i+1) % kwargs.get('update_freq') == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        acc = compute_acc(batch_outputs, batch_inputs)[0]
        
        meter['loss'].update(losses.item(), batch_size)
        meter['dice'].update(acc.item(), batch_size)
        meter['lr'].reset()
        meter['lr'].update(scheduler.get_last_lr()[0], 1)
        tqdm_bar.set_postfix({key: val.item() for key, val in meter.items()})

    return meter['loss'].item(), meter['dice'].item()


def validation_step(ep, model, valid_loader, criterion, device, **kwargs):
    model.eval()
    meter = init_meter('valid')
    tqdm_bar = tqdm(valid_loader, desc=f'Validation {ep}')
    for batch_inputs in tqdm_bar:
        batch_size = batch_inputs['image'].size(0)  
        batch_inputs = label_downsample(batch_inputs, kwargs.get('down_times'))
        batch_inputs = dict_to_device(batch_inputs,
                                      keys=kwargs.get('device_keys'),
                                      device=device)
        
        batch_outputs = model.predict(batch_inputs)

        batch_inputs  = crop_from_pad(batch_inputs, ['image', 'label', 'label_0'])
        batch_outputs = crop_from_pad(batch_outputs, ['stage_0'])
        
        losses = criterion(batch_outputs, batch_inputs)
        acc = compute_acc(batch_outputs, batch_inputs)[0]

        meter['loss'].update(losses.item(), batch_size)
        meter['dice'].update(acc.item(), batch_size)
        tqdm_bar.set_postfix({key: val.item() for key, val in meter.items()})

    return meter['loss'].item(), meter['dice'].item()


def trainer(args):
    cur_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    save_root = osp.join(args.TRAIN.SAVE_ROOT, cur_time)
    weightPath = osp.join(save_root, 'model_weights')
    os.makedirs(weightPath, exist_ok=True)
    WriteYaml(osp.join(save_root, 'config.yaml'), args)

    device = torch.device(f"cuda:{args.BASIC.DEVICE[0]}"
                          if torch.cuda.is_available() else 'cpu')
    model = get_model(**args)
    model.data_parallel(device_ids=args.BASIC.DEVICE)
    model = model.to(device)
    optimizer = get_optimizer(model, **args.OPT)
    scheduler = get_scheduler(optimizer, **args.OPT)
    criterion = get_criterion(**args.LOSS)

    TrainDataset = stasDataset(args, stage='train')
    TrainLoader = DataLoader(dataset=TrainDataset,
                             batch_size=args.OPT.SUB_BATCH_SIZE,
                             shuffle=True,
                             num_workers=args.BASIC.NUM_WORKER,
                             collate_fn=stasCollateFn,
                             pin_memory=True)
    
    ValidDataset = stasDataset(args, stage='valid')
    ValidLoader = DataLoader(dataset=ValidDataset,
                             batch_size=1,
                             collate_fn=stasCollateFn,
                             pin_memory=True)

    logger = Logger()
    device_keys = ['image', 'label'] + args.LOSS.LABEL_KEY
    down_times = args.MODEL.DECODER.DEEP_SUPERVISION_DEPTH
    update_freq = args.OPT.UPDATE_FREQ
    save_model(weightPath, model, 0, 0, args.TRAIN.WEIGHT_NUM)

    for ep in range(1, args.OPT.EPOCHS+1):
        criterion.update_weights(cur_ep=ep, total_ep=args.OPT.EPOCHS)
        #update_loader(TrainLoader, cur_ep=ep, total_ep=args.OPT.EPOCHS)
        
        train_loss, train_acc = training_step(
            ep, model, TrainLoader, criterion, optimizer, scheduler, device,
            device_keys=device_keys, down_times=down_times, update_freq=update_freq
        )
        logger.add(
            epoch=ep,
            type='train',
            loss=train_loss,
            dice=train_acc,
            lr=scheduler.get_last_lr()[0]
        )
        
        valid_loss, valid_acc = validation_step(
            ep, model, ValidLoader, criterion, device,
            device_keys=device_keys, down_times=down_times
        )
        logger.add(
            epoch=ep,
            type='valid',
            loss=valid_loss,
            dice=valid_acc,
            lr=scheduler.get_last_lr()[0]
        )

        save_model(weightPath, model, ep, valid_acc, args.TRAIN.WEIGHT_NUM+1)
        logger.save(os.path.join(save_root, 'record.csv'))

    return 
