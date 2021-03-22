_base_="../base-domain_net_clipart-domain_net_sketch-config.py"

# this will merge with the parent
model=dict(pretrained='work_dirs/hpt-pretrain/domain_net_clipart/moco_v2_800ep_basetrain/5000-iters/moco_v2_800ep_basetrain-domain_net_clipart_5000it.pth')

# epoch related
total_iters=50000
checkpoint_config = dict(interval=total_iters)
