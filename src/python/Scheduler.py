import torch
from torch.optim.lr_scheduler import _LRScheduler, StepLR, CosineAnnealingLR
from torch.optim import Optimizer, Adam
import matplotlib.pyplot as plt

__all__ = ["WarmupScheduler"]

class WarmupScheduler(_LRScheduler):
    '''
    warm up scheduler
    reference: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py)

    Args:
        optimizer (Optimizer): Wrapped optimizer
        warm_epoch: number of epoch that used to warm up
        scheduler: after warm up, apply this scheduler
    '''
    def __init__(self, optimizer, warm_steps, start_lr,
                 origin_scheduler, last_epoch=-1):
        assert (warm_steps >= 0)
        assert isinstance(optimizer, Optimizer)
        assert isinstance(origin_scheduler, _LRScheduler)

        self.optimizer = optimizer
        self.scheduler = origin_scheduler
        self.warm_steps = warm_steps
        self.warmed = False
        self.start_lr = start_lr
        
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        if self.last_epoch <= self.warm_steps:
            return [base_lr
                    for base_lr in self.base_lrs]
        else:
        '''
        if self.last_epoch < self.warm_steps:
            return [base_lr * (self.last_epoch / self.warm_steps)
                    for base_lr in self.base_lrs]
        else:
            self.warmed = True
            return self.scheduler.get_last_lr()

    def step(self, epoch=None):
        if not self.warmed:
            return super(WarmupScheduler, self).step(epoch)
        else:
            if epoch is None:
                self.scheduler.step(None)
            else:
                self.scheduler.step(epoch - self.warm_steps)
            self._last_lr = self.scheduler.get_last_lr()
        
if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = Adam(model, lr=5e-4)

    IT_PER_EP = 92
    EPOCH_IT  = 150 * IT_PER_EP
    WARMUP_IT = 3 * IT_PER_EP
    
    #steplr = StepLR(optimizer, step_size=20, gamma=0.5)
    cosann = CosineAnnealingLR(optimizer, T_max=EPOCH_IT-WARMUP_IT, eta_min=1e-6)
    warmup = WarmupScheduler(optimizer,
                             warm_steps=WARMUP_IT,
                             start_lr=0,
                             origin_scheduler=cosann)

    optimizer.zero_grad()
    optimizer.step()
    LR_account = []
    for iteration in range(EPOCH_IT):
        warmup.step()
        #print(iteration, warmup.get_last_lr())
        LR_account.append(warmup.get_last_lr()[0])
        optimizer.step()
    plt.plot(LR_account)
    plt.grid()
    plt.show()
