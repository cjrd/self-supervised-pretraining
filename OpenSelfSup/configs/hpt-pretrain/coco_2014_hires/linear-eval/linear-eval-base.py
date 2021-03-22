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
        num_classes=80,
        use_bce_loss=True
    )
)

# dataset settings
data_source_cfg = dict(
    type="ImageListMultihead",
    memcached=False,
    mclient_path='/not/used',
    # this will be ignored if type != ImageListMultihead
    
)

# used to trian the linear classifier
data_train_list = "coco_2014/meta/train.txt"
data_train_root = ""

# used for val (ie picking the final model)
data_val_list = "coco_2014/meta/val.txt"
data_val_root = ""

# used for testing evaluation: we've never seen this data before (not even during pretraining)
data_test_list = "coco_2014/meta/test.txt"
data_test_root = ""

dataset_type = "AUROCDataset"
img_norm_cfg = dict(mean=[0.4702, 0.4470, 0.4076], std=[0.2785, 0.2740, 0.2889])
train_pipeline = [
    dict(type='RandomResizedCrop', size=640),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=640),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    batch_size=512,
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
        workers_per_gpu=4,
        eval_param=dict()),
    dict(
        name="test",
        type='ValidateHook',
        by_epoch=False,
        dataset=data['test'],
        initial=True,
        interval=100,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict())
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
