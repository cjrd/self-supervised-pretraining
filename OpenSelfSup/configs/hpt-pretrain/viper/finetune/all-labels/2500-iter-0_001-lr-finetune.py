_base_ = "finetune-eval-base.py"

# dataset settings
data_source_cfg = dict(
    type="ImageListMultihead",
    memcached=False,
    mclient_path='/no/matter',
    # this will be ignored if type != ImageListMultihead
    class_map={'0': '0', '-1': '0', '': '0', '1': '1'}
)


data_train_list = "data/viper/meta/train_multi_label.txt"
data_train_root = 'data/viper'

data_val_list = "data/viper/meta/val_multi_label.txt"
data_val_root = 'data/viper'

data_test_list = "data/viper/meta/test_multi_label.txt"
data_test_root = 'data/viper'

dataset_type = "AUROCDataset"
img_norm_cfg = dict(mean=[0.3485, 0.3422, 0.3395], std=[0.2449, 0.2408, 0.2390])

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
    batch_size=64, # x4 from update_interval
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
        by_epoch=False,
        initial=False,
        interval=25,
        imgs_per_gpu=32,
        workers_per_gpu=5,
        eval_param=dict()),
    dict(
        name="test",
        type='ValidateHook',
        dataset=data['test'],
        by_epoch=False,
        initial=False,
        interval=25,
        imgs_per_gpu=32,
        workers_per_gpu=5,
        eval_param=dict()),
]

by_iter =True

# learning policy
lr_config = dict(
    by_epoch=False,
    policy='step',
    step=[833,1667],
    gamma=0.1  # multiply LR by this number at each step
)

# momentum and weight decay from VTAB and IDRL
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.,
                 paramwise_options={'\Ahead.': dict(lr_mult=100)})


# runtime settings
# total iters or total epochs
total_iters=2500
checkpoint_config = dict(interval=2500)

log_config = dict(
    interval=1,
    by_epoch=False,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])

optimizer_config = dict(update_interval=4)
