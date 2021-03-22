_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=45))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/no/matter')
data_train_list = 'data/resisc45/meta/train_labeled_800.txt'
data_train_root = 'data/resisc45/train'
data_test_list = 'data/resisc45/meta/val_labeled.txt'
data_test_root = 'data/resisc45/val'
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
        initial=False,
        interval=20,
        imgs_per_gpu=32,
        workers_per_gpu=2,
        eval_param=dict(topk=(1, 5)))
]
# learning policy
lr_config = dict(policy='step', step=[200, 400], gamma=0.2)
checkpoint_config = dict(interval=600)
# runtime settings
total_epochs = 600
