_base_="../base-chest_xray_kids-config.py"

# this will merge with the parent

# epoch related
total_iters=50000
checkpoint_config = dict(interval=total_iters)
checkpoint_config = dict(interval=total_iters//2)
