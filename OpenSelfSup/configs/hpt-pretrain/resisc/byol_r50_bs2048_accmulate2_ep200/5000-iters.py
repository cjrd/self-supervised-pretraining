_base_="../byol-base-resisc-config.py"

# this will merge with the parent
model=dict(pretrained='data/basetrain_chkpts/byol_r50_bs2048_accmulate2_ep200.pth')

# epoch related
total_iters=5000*2
checkpoint_config = dict(interval=total_iters)
