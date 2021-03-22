_base_ = '../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained="/rscratch/cjrd/basetraining/OpenSelfSup/work_dirs/selfsup/simclr/r50_bs256_ep20_resisc_baseline_correct_pixel_norm/epoch_20_resisc_baseline_correct_pixel_norm_backbone.pth",
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=21))
# dataset settings
data_source_cfg = dict(
    type='ImageList',
    memcached=False,
    mclient_path='/no/matter')
data_train_list = 'data/ucmerced/meta/train_labeled.txt'
data_train_root = 'data/ucmerced'
data_test_list = 'data/ucmerced/meta/val_labeled.txt'
data_test_root = 'data/ucmerced'
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.4842, 0.4901, 0.4505], std=[0.2180, 0.2021, 0.1958])
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,  # total 256
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=10,
        imgs_per_gpu=64,
        workers_per_gpu=2,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# learning policy
# lr_config = dict(policy='step', step=[30, 60, 90])
lr_config = dict(
    policy='step',
    step=[30, 60, 80, 90],
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 90
