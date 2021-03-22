_base_="../base-${shortname}-config.py"

# this will merge with the parent
model=dict(pretrained='${pretrained}')

# epoch related
total_iters=${iter}
checkpoint_config = dict(interval=total_iters)

