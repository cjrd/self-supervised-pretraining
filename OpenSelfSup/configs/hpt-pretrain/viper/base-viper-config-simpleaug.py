_base_ = '../../base.py'
# model settings
model = dict(
    type='MOCO',
    pretrained=None,
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/not/used')

data_train_list = "data/viper/meta/train_val.txt"
data_train_root = "data/viper"

dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.3485, 0.3422, 0.3395], std=[0.2449, 0.2408, 0.2390])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    batch_size=256,
    workers_per_gpu=5,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# cjrd added this flag, since OSS didn't support training by iters(?)
by_iter = True

log_config = dict(
    interval=25,
    by_epoch=False,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
