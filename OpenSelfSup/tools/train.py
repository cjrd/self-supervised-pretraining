from __future__ import division
import argparse
import importlib
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from openselfsup import __version__
from openselfsup.apis import set_random_seed, train_model
from openselfsup.datasets import build_dataset
from openselfsup.models import build_model
from openselfsup.utils import collect_env, get_root_logger, traverse_replace


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--pretrained', default=None, help='pretrained model file')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='debug (one gpu - disable DistributedDataParallel). With this flag, you can set breakpoints')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    print(f"Using num gpus: {torch.cuda.device_count()}")
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if not args.debug:
            assert cfg.model.type not in \
                ['DeepCluster', 'MOCO', 'SimCLR', 'ODC', 'NPID'], \
                "{} does not support non-dist training unless debugging (use --debug flag).".format(
                    cfg.model.type)
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'train_{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    if args.pretrained is not None:
        assert isinstance(args.pretrained, str)
        cfg.model.pretrained = args.pretrained
    model = build_model(cfg.model)
    if args.debug:
        # TODO(cjrd) fix this hardcoding?
        logger.info(
            "DEBUGGING enabled, setting batch size to 64 to allow 1 gpu debugging")
        cfg.data.batch_size = 64
        model.set_debug()

    datasets = [build_dataset(cfg.data.train)]
    assert len(cfg.workflow) == 1, "Validation is called by hook."
    if cfg.checkpoint_config is not None:
        # save openselfsup version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            openselfsup_version=__version__, config=cfg.text)
    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta,
        debug=args.debug)


if __name__ == '__main__':
    main()
