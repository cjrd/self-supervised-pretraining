_base_="../base-chexpert_rm_color-config.py"

# this will merge with the parent
model=dict(pretrained='data/basetrain_chkpts/imagenet_r50_supervised.pth')

# epoch related
total_iters=50000
checkpoint_config = dict(interval=total_iters)
