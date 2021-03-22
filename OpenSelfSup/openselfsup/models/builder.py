from torch import nn
from torch import randn

from openselfsup.utils import build_from_cfg
from .registry import (BACKBONES, MODELS, NECKS, HEADS, MEMORIES, LOSSES)


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Default: None.

    Returns:
        nn.Module: A built nn module.
    """

    # ugly hack to __automagically__ set the neck input
    if cfg.get("neck") and cfg.get("neck").get("auto_channels"):
        del cfg["neck"]["auto_channels"]
        # build the backbone to obtain the number of channels
        bbone = build_backbone(cfg['backbone'])
        x = randn(1, cfg['backbone']['in_channels'], 224, 224) # 224 doesn't really matter here
        outp = bbone(x)
        if isinstance(outp, tuple):
            outp = outp[0]
        outsize = outp.shape[1]
        del bbone, x
        cfg["neck"]["in_channels"] = outsize
        if cfg['neck']['type'].find('NonLinear') > -1:
            cfg["neck"]["hid_channels"] = outsize
    
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_memory(cfg):
    """Build memory."""
    return build(cfg, MEMORIES)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_model(cfg):
    """Build model."""
    return build(cfg, MODELS)
