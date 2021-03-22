_base_="../base-bdd-config.py"

# this will merge with the parent
model=dict(pretrained='/home/cjrd/dev/base-training/OpenSelfSup/data/basetrain_chkpts/moco_v2_800ep.pth')

# epoch related
total_iters=5000
checkpoint_config = dict(interval=total_iters)
