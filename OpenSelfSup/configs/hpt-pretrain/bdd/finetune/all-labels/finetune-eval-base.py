train_cfg = {}
test_cfg = {}
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb
# yapf:disable
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        # TODO(cjrd) should we be using BN here???
        norm_cfg=dict(type='BN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=6,
        
    )
)
prefetch=False
