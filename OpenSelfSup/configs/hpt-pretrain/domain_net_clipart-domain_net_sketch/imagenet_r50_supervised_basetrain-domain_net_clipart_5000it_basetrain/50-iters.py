_base_="../base-domain_net_clipart-domain_net_sketch-config.py"

# this will merge with the parent
model=dict(pretrained='work_dirs/hpt-pretrain/domain_net_clipart/imagenet_r50_supervised_basetrain/5000-iters/imagenet_r50_supervised_basetrain-domain_net_clipart_5000it.pth')

# epoch related
total_iters=50
checkpoint_config = dict(interval=total_iters)
