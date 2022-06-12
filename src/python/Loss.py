import math

import monai
import torch
import torch.nn as nn

__all__ = ["MonaiLoss"]

class MonaiLoss(nn.Module):
    def __init__(self, loss_ctor, **kwargs):
        super(MonaiLoss, self).__init__()
        predKeys  = kwargs.get('PRED_KEY')
        labelKeys = kwargs.get('LABEL_KEY')
        actKeys   = kwargs.get('ACT_KEY')
        weights   = kwargs.get('WEIGHT')
        assert (len(predKeys) == len(labelKeys))
        assert (len(predKeys) == len(weights))

        self.num_terms = len(weights)
        self.init_weights = weights
        self.weights = weights
        self.predKeys = predKeys
        self.labelKeys = labelKeys
        self.actKeys = actKeys

        self.sigmoidCrit = loss_ctor(to_onehot_y=False)
        self.softMaxCrit = loss_ctor(to_onehot_y=True)
        
    def forward(self, predDict, labelDict):
        Loss = []
        for pk, lk, act, w in zip(self.predKeys, self.labelKeys, self.actKeys, self.weights):
            if act == 'sigmoid':
                Loss.append( w * self.sigmoidCrit(predDict[pk], labelDict[lk]) )
            elif act == 'softmax':
                Loss.append( w * self.softMaxCrit(predDict[pk], labelDict[lk]) )
        return sum(Loss)

    def update_weights(self, cur_ep, total_ep):
        for i in range(1, self.num_terms):
            #self.weights[i] = self.init_weights[i] * (1 - cur_ep / total_ep)
            self.weights[i] = self.init_weights[i] * math.cos(math.pi * cur_ep * 0.5 / total_ep)
        return

