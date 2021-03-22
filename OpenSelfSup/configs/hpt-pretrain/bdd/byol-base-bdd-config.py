import copy
_base_ = '../../base.py'
# model settings
model = dict(
    type='BYOL',
    pretrained=None,
    base_momentum=0.99,
    pre_conv=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        sync_bn=True,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckSimCLR',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, num_layers=2, sync_bn=True,
                             with_bias=True, with_last_bn=False, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/not/used')

data_train_list = "data/bdd/meta/train_val.txt"
data_train_root = "data/bdd"

dataset_type = 'BYOLDataset'
img_norm_cfg = dict(mean=[0.2789, 0.2929, 0.2902], std=[0.2474, 0.2653, 0.2761])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3), # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline2[4]['p'] = 0.1 # gaussian blur
train_pipeline2[5]['p'] = 0.2 # solarization

data = dict(
    imgs_per_gpu=64,  # total 32*8=256
    batch_size=128,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    ))
# additional hooks
update_interval = 2  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]
# optimizer
optimizer = dict(type='LARS', lr=4.8/16, weight_decay=0.000001, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})
# apex
use_fp16 = False
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=False)

# cjrd added this flag, since OSS didn't support training by iters(?)
by_iter = True

log_config = dict(
    interval=25,
    by_epoch=False,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
