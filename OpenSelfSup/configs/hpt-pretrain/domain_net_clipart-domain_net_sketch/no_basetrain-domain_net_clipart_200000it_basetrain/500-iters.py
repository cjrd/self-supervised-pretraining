_base_="../base-domain_net_clipart-domain_net_sketch-config.py"

# this will merge with the parent
model=dict(pretrained='work_dirs/hpt-pretrain/domain_net_clipart/no_basetrain/200000-iters/no_basetrain-domain_net_clipart_200000it.pth')

# epoch related
total_iters=500
checkpoint_config = dict(interval=total_iters)
