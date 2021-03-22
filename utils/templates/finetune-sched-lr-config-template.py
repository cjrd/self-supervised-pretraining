_base_ = "finetune-eval-base.py"

# dataset settings
data_source_cfg = dict(
    type="${image_head_class_type}",
    memcached=False,
    mclient_path='/no/matter',
    # this will be ignored if type != ImageListMultihead
    ${class_map}
)


data_train_list = "${ft_train_list_path}"
data_train_root = '${base_data_path}'

data_val_list = "${ft_val_list_path}"
data_val_root = '${base_data_path}'

data_test_list = "${ft_test_list_path}"
data_test_root = '${base_data_path}'

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
    batch_size=64, # x4 from update_interval
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


custom_hooks = [
    dict(
        name="val",
        type='ValidateHook',
        dataset=data['val'],
        by_epoch=${by_epoch},
        initial=False,
        interval=${val_interval},
        imgs_per_gpu=32,
        workers_per_gpu=${workers_per_gpu},
        eval_param=${eval_params}),
    dict(
        name="test",
        type='ValidateHook',
        dataset=data['test'],
        by_epoch=${by_epoch},
        initial=False,
        interval=${val_interval},
        imgs_per_gpu=32,
        workers_per_gpu=${workers_per_gpu},
        eval_param=${eval_params}),
]

by_iter =${by_iter}

# learning policy
lr_config = dict(
    by_epoch=${by_epoch},
    policy='step',
    step=[${lr_steps}],
    gamma=0.1  # multiply LR by this number at each step
)

# momentum and weight decay from VTAB and IDRL
optimizer = dict(type='SGD', lr=${ft_lr}, momentum=0.9, weight_decay=0.,
                 paramwise_options={'\Ahead.': dict(lr_mult=100)})


# runtime settings
# total iters or total epochs
${total_line}
checkpoint_config = dict(interval=${total_val})

log_config = dict(
    interval=${log_interval},
    by_epoch=${by_epoch},
    hooks=[
        dict(type='TextLoggerHook', by_epoch=${by_epoch}),
        dict(type='TensorboardLoggerHook', by_epoch=${by_epoch})
    ])

optimizer_config = dict(update_interval=4)

