_base_="../base-resisc-bn-config.py"

# this will merge with the parent
model=dict(pretrained='data/basetrain_chkpts/imagenet_r50_supervised.pth')

# epoch related
total_iters=500
checkpoint_config = dict(interval=total_iters)
