_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=45))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/resisc45/meta/train_labeled.txt'
data_train_root = 'data/resisc45'
data_test_list = 'data/resisc45/meta/val_labeled.txt'
data_test_root = 'data/resisc45'
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.368, 0.381, 0.3436], std=[0.2035, 0.1854, 0.1849])
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
    imgs_per_gpu=128,  # total 128*4=512, 4GPU linear cls
    workers_per_gpu=5,
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
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)
# learning policy
lr_config = dict(policy='step', step=[60, 80])
checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 100
