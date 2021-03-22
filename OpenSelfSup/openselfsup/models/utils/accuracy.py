import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res

def auroc(pred, target):
    outAUROC = torch.zeros(pred.shape[1])
    
    datanpGT = target.detach().cpu().numpy()
    datanpPRED = pred.detach().cpu().numpy()
    
    for i in range(pred.shape[1]):
        try:
            outAUROC[i] = roc_auc_score(datanpGT[:, i], datanpPRED[:, i])
        except ValueError:
            pass
    return outAUROC.cuda()

class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)
