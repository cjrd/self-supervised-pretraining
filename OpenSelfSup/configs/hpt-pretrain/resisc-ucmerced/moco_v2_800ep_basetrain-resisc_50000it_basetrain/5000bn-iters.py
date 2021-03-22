_base_="../base-resisc-ucmerced-config.py"

# this will merge with the parent

# epoch related
total_iters=5000
checkpoint_config = dict(interval=total_iters)

model = dict(
    pretrained='work_dirs/hpt-pretrain/resisc/moco_v2_800ep_basetrain/50000-iters/moco_v2_800ep_basetrain-resisc_50000it.pth',
    backbone=dict(
        norm_train=True,
        frozen_stages=4,
    )
)
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
