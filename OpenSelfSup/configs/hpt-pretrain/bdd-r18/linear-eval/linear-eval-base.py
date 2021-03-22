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
        num_classes=6,
        
    )
)

# dataset settings
data_source_cfg = dict(
    type="ImageNet",
    memcached=False,
    mclient_path='/not/used',
    # this will be ignored if type != ImageListMultihead
    
)

# used to trian the linear classifier
data_train_list = "data/bdd/meta/train_weather_labeled.txt"
data_train_root = "data/bdd"

# used for val (ie picking the final model)
data_val_list = "data/bdd/meta/val_weather_labeled.txt"
data_val_root = "data/bdd"

# used for testing evaluation: we've never seen this data before (not even during pretraining)
data_test_list = "data/bdd/meta/test_weather_labeled.txt"
data_test_root = "data/bdd"

dataset_type = "ClassificationDataset"
img_norm_cfg = dict(mean=[0.2789, 0.2929, 0.2902], std=[0.2474, 0.2653, 0.2761])
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
    batch_size=512,
    workers_per_gpu=6,
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

# additional hooks
custom_hooks = [
    dict(
        name="val",
        type='ValidateHook',
        dataset=data['val'],
        by_epoch=False,
        initial=True,
        interval=100,
        imgs_per_gpu=128,
        workers_per_gpu=6,
        eval_param=dict(topk=(1,5))),
    dict(
        name="test",
        type='ValidateHook',
        by_epoch=False,
        dataset=data['test'],
        initial=True,
        interval=100,
        imgs_per_gpu=128,
        workers_per_gpu=6,
        eval_param=dict(topk=(1,5)))
]

# learning policy
lr_config = dict(
    by_epoch=False,
    policy='step',
    step=[1651,3333])
checkpoint_config = dict(interval=5000)

# runtime settings
total_iters = 5000
checkpoint_config = dict(interval=total_iters)

# cjrd added this flag, since OSS didn't support training by iters(?)
by_iter = True

log_config = dict(
    interval=10,
    by_epoch=False,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
