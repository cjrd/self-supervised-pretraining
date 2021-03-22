_base_="../base-chexpert-chest_xray_kids-config.py"

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

optimizer = dict(type='SGD', lr=0.003, weight_decay=0.0001, momentum=0.9)
