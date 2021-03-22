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
        num_classes=${num_classes},
        ${bce_string}
    )
)

# dataset settings
data_source_cfg = dict(
    type="${image_head_class_type}",
    memcached=False,
    mclient_path='/not/used',
    # this will be ignored if type != ImageListMultihead
    ${class_map}
)

# used to trian the linear classifier
data_train_list = "${train_list_path}"
data_train_root = "${base_data_path}"

# used for val (ie picking the final model)
data_val_list = "${val_list_path}"
data_val_root = "${base_data_path}"

# used for testing evaluation: we've never seen this data before (not even during pretraining)
data_test_list = "${test_list_path}"
data_test_root = "${base_data_path}"

dataset_type = "${dataset_type}"
img_norm_cfg = dict(mean=[${pixel_means}], std=[${pixel_stds}])
train_pipeline = [
    dict(type='RandomResizedCrop', size=${crop_size}),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=${test_precrop_size}),
    dict(type='CenterCrop', size=${crop_size}),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    batch_size=512,
    workers_per_gpu=${workers_per_gpu},
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
        initial=False,
        interval=100,
        imgs_per_gpu=128,
        workers_per_gpu=${workers_per_gpu},
        eval_param=${eval_params}),
    dict(
        name="test",
        type='ValidateHook',
        by_epoch=False,
        dataset=data['test'],
        initial=False,
        interval=100,
        imgs_per_gpu=128,
        workers_per_gpu=${workers_per_gpu},
        eval_param=${eval_params})
]

# learning policy
lr_config = dict(
    by_epoch=False,
    policy='step',
    step=[${linear_lr_drop_iters}])
checkpoint_config = dict(interval=${linear_iters})

# runtime settings
total_iters = ${linear_iters}
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
