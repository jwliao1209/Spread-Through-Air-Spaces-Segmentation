import torch
import monai

from .model import EfficientUNet
from .Loss import MonaiLoss
from .Scheduler import WarmupScheduler

__all__ = ["get_model", "get_optimizer", "get_scheduler", "get_criterion"]

def get_criterion(**kwargs):
    Losses = {
        'DL':   monai.losses.DiceLoss,
        'GDL':  monai.losses.GeneralizedDiceLoss,
        'DCEL': monai.losses.DiceCELoss,
        'DFL':  monai.losses.DiceFocalLoss
    }
    loss_ctor = Losses[kwargs.get('NAME')]
    return MonaiLoss(loss_ctor, **kwargs)

def get_model(**kwargs):
    Model = {
        'unet': EfficientUNet
    }
    model = Model[kwargs.get('MODEL').get('NAME')](**kwargs)
    
    return model

def get_optimizer(model, **kwargs):
    Optimizer = {
        'sgd'  : torch.optim.SGD,
        'adam' : torch.optim.Adam,
        'adamw': torch.optim.AdamW
    }
    optimizer = Optimizer[kwargs.get('OPTIMIZER')](
        params=model.parameters(),
        lr=kwargs.get('LEARNING_RATE'),
        weight_decay=kwargs.get('WEIGHT_DECAY')
    )
    
    return optimizer

def get_scheduler(optimizer, **kwargs):
    Scheduler = {
        'step': torch.optim.lr_scheduler.StepLR(
              optimizer=optimizer,
              step_size=kwargs.get('STEP_SIZE'),
              gamma=kwargs.get('GAMMA')
        ),
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=kwargs.get('T_MAX'),
            eta_min=kwargs.get('ETA_MIN')
        )
    }
    scheduler = Scheduler[kwargs.get('SCHEDULER')]
    
    warmup = WarmupScheduler(optimizer=optimizer,
                             warm_steps=kwargs.get('WARMUP_IT'),
                             start_lr=0,
                             origin_scheduler=scheduler,
                             last_epoch=-1)
    return warmup
