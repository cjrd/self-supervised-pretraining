_base_="finetune-eval-base.py"

data = dict()
custom_hooks = [
    dict(
        name="val",
        type='ValidateHook',
        dataset=data['val'],
        by_epoch=True,
        initial=False,
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5))),
    dict(
        name="test",
        type='ValidateHook',
        dataset=data['test'],
        by_epoch=True,
        initial=False,
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]

by_iter=False

# learning policy
lr_config = dict(
    by_epoch=True,
    policy='step',
    step=[30,60],
    gamma=0.1, # multiply LR by this number at each step
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001, ## starting LR is warmup_ratio * lr
    warmup_by_epoch=True    
)

# momentum and weight decay from VTAB and IDRL
optimizer = dict(type='SGD', lr=0.01., momentum=0.9, weight_decay=0.)


# runtime settings
# total iters or total epochs
total_epochs=90
checkpoint_config = dict(interval=90)
optimizer_config = dict(update_interval=4)
