_base_="../base-chexpert-r18-config.py"

# this will merge with the parent

# epoch related
total_iters=5000
checkpoint_config = dict(interval=total_iters)
checkpoint_config = dict(interval=total_iters//2)
