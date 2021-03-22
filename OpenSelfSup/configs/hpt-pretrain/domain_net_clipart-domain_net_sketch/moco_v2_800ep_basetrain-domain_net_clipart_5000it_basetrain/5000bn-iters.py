_base_="../base-domain_net_clipart-domain_net_sketch-config.py"

# this will merge with the parent

# epoch related
total_iters=5000
checkpoint_config = dict(interval=total_iters)

model = dict(
    pretrained='work_dirs/hpt-pretrain/domain_net_clipart/moco_v2_800ep_basetrain/5000-iters/moco_v2_800ep_basetrain-domain_net_clipart_5000it.pth',
    backbone=dict(
        norm_train=True,
        frozen_stages=4,
    )
)
optimizer = dict(type='SGD', lr=0.1, weight_decay=0.0001, momentum=0.9)
