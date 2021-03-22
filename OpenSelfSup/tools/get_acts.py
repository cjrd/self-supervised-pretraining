import argparse
import importlib
import os
import os.path as osp
import time
from collections import defaultdict
from functools import partial

import mmcv
import torch
from torch import nn
import numpy as np
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model
from openselfsup.utils import (get_root_logger, dist_forward_collect, 
                               nondist_forward_collect, traverse_replace)


def single_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results


def multi_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_forward_collect(func, data_loader, rank,
                                   len(data_loader.dataset))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    parser.add_argument('--grab-conv',dest='layer_type', action='store_const', 
            default=nn.Linear, const=nn.Conv2d,
        help='save conv activations instead of linear')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'test_{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model)

    activations = defaultdict(list)
    #idea from gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
    if args.layer_type == nn.Linear: #can save all activations
        def save_activation(name, mod, inp, out):
            activations[name].append(out.cpu())
    else:
        def save_activation(name, mod, inp, out):
            activations[name] = [out.cpu()]
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    for name, m in model.named_modules():
        if type(m) == args.layer_type:
            m.register_forward_hook(partial(save_activation, name))


    if not distributed:  
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        raise NotImplementedError("Distributed Data Parallel does not register hooks.")
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)  # dict{key: np.ndarray}


    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    act_file = osp.join(cfg.work_dir, "model_acts")
    np.savez(act_file, **activations)
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     for name, val in outputs.items():
    #         dataset.evaluate(
    #             torch.from_numpy(val), name, logger, topk=(1, 5))


if __name__ == '__main__':
    main()
