_base_="../base-coco_2014-config-simpleaug.py"

# this will merge with the parent
model=dict(pretrained='data/basetrain_chkpts/moco_v2_800ep.pth')

# epoch related
total_iters=5000
checkpoint_config = dict(interval=total_iters)
model = dict(
    pretrained='data/basetrain_chkpts/moco_v2_800ep.pth',
    backbone=dict(
	norm_train=True,
	frozen_stages=4,
    )
)
optimizer = dict(type='SGD', lr=0.1, weight_decay=0.0001, momentum=0.9)
