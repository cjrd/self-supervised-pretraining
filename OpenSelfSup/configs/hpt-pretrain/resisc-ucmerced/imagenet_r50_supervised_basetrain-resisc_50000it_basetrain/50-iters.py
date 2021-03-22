_base_="../base-resisc-ucmerced-config.py"

# this will merge with the parent
model=dict(pretrained='work_dirs/hpt-pretrain/resisc/imagenet_r50_supervised_basetrain/50000-iters/imagenet_r50_supervised_basetrain-resisc_50000it.pth')

# epoch related
total_iters=50
checkpoint_config = dict(interval=total_iters)
