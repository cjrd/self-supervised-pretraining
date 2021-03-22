_base_="../base-pascal-config.py"

# this will merge with the parent
model=dict(pretrained='work_dirs/hpt-pretrain/coco_2014/moco_v2_800ep_basetrain/50000-iters/final_backbone.pth')

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
