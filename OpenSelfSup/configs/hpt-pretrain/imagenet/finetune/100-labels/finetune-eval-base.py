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
        # TODO(cjrd) should we be using BN here???
        norm_cfg=dict(type='BN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=10))

# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/no/matter')

data_train_list = "data/imagenet/meta/train_100.txt"
data_train_root = 'data/imagenet'

data_val_list = "data/imagenet/meta/val_100.txt"
data_val_root =  'data/imagenet'

data_test_list = "data/imagenet/meta/test_100.txt"
data_test_root =  'data/imagenet'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.5,0.6,0.7], std=[0.1,0.2,0.3])

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
    imgs_per_gpu=128,  # total 512
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_val_list, root=data_val_root, **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))
prefetch=False
