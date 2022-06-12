import torch

__all__ = ["compute_acc", "compute_metric", "img_to_binary"]

def img_to_binary(img, img_type):
    assert img_type in ['softmax', 'sigmoid']

    if img_type == 'softmax':
        return torch.argmax(img, dim=1)
    if img_type == 'sigmoid':
        return img.squeeze(1) > 0.5

def compute_metric(preds, targets, smooth_eps=1e-7):
    targetsP = (targets > 0.5).float()
    targetsN = (targets <= 0.5).float()
    predsP = (preds > 0.5).float()
    predsN = (preds <= 0.5).float()

    TP = torch.sum(targetsP * predsP, dim=(1,2))
    TN = torch.sum(targetsN * predsN, dim=(1,2))
    FP = torch.sum(targetsN * predsP, dim=(1,2))
    FN = torch.sum(targetsP * predsN, dim=(1,2))

    dice = 2 * TP / (FP + FN + 2 * TP + smooth_eps)
    recall = TP / (TP + FN + smooth_eps)
    precision = TP / (TP + FP + smooth_eps)
    return dice.mean(), recall.mean(), precision.mean()
    

def compute_acc(outDict, inDict):
    if (outDict['stage_0'].size(1) > 1):
        preds = img_to_binary(outDict['stage_0'], 'softmax')
    else:
        preds = img_to_binary(outDict['stage_0'], 'sigmoid')
    return compute_metric(preds, inDict['label'].squeeze(1))
