_base_="../base-resisc-ucmerced-config.py"

# this will merge with the parent
model=dict(pretrained='work_dirs/hpt-pretrain/resisc/no_basetrain/200000-iters/no_basetrain-resisc_200000it.pth')

# epoch related
total_iters=5000
checkpoint_config = dict(interval=total_iters)
