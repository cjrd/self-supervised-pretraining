import torch

from openselfsup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from ..models.utils import auroc

@DATASETS.register_module
class AUROCDataset(BaseDataset):
    """Dataset for computing auroc.
    """

    def __init__(self, data_source, pipeline):
        super(AUROCDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        return dict(img=img, gt_label=target)

    def evaluate(self, scores, keyword, logger=None):
        
        eval_res = {}
        target = torch.cat(self.data_source.labels).reshape(scores.shape)
        
        aurocs = auroc(scores, target).cpu()
        for ii in range(len(aurocs)):
            res = aurocs[ii].item()
            eval_res["auroc_{}".format(ii)] = res
            if logger is not None and logger != 'silent':
                print_log(
                    "auroc_{}: {:.03f}".format(ii, res),
                    logger=logger)
        return eval_res
