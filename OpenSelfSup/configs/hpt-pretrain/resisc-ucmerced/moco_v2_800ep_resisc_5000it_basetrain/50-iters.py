_base_="../base-resisc-ucmerced-config.py"

# this will merge with the parent
model=dict(pretrained='work_dirs/hpt-pretrain/resisc/moco_v2_800ep_basetrain/5000-iters/moco_v2_800ep_resisc_5000it.pth')

# epoch related
total_iters=50
checkpoint_config = dict(interval=total_iters)
