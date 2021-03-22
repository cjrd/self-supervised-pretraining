_base_="../base-bdd-r18-config.py"

# this will merge with the parent
model=dict(pretrained='data/basetrain_chkpts/mocoV2_200ep_resnet18.pth')

# epoch related
total_iters=50000
checkpoint_config = dict(interval=total_iters)
