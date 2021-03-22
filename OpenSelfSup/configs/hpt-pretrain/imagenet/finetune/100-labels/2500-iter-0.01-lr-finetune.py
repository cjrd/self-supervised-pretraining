_base_="finetune-eval-base.py"

data = dict()
custom_hooks = [
    dict(
        name="val",
        type='ValidateHook',
        dataset=data['val'],
        by_epoch=False,
        initial=False,
        interval=100,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5))),
    dict(
        name="test",
        type='ValidateHook',
        dataset=data['test'],
        by_epoch=False,
        initial=False,
        interval=100,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]

by_iter=True

# learning policy
lr_config = dict(
    by_epoch=False,
    policy='step',
    step=[833,1667],
    gamma=0.1, # multiply LR by this number at each step
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.0001, ## starting LR is warmup_ratio * lr
    warmup_by_epoch=False    
)

# momentum and weight decay from VTAB and IDRL
optimizer = dict(type='SGD', lr=0.01., momentum=0.9, weight_decay=0.)


# runtime settings
# total iters or total epochs
total_iters=2500
checkpoint_config = dict(interval=10000)
optimizer_config = dict(update_interval=4)
