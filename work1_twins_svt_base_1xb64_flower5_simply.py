_base_ = [
    '../_base_/models/twins_svt_base.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]



paramwise_cfg = dict(_delete=True, norm_decay_mult=0.0, bias_decay_mult=0.0)

# for batch in each gpu is 64, 1 gpu
# lr = 5e-4 * 16 * 1 / 512 
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 64  / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=True,
    min_lr_ratio=1e-3,
    warmup='linear',
    warmup_ratio=1e-4,
    warmup_iters=5,
    warmup_by_epoch=True)


data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        ann_file = 'data/train.txt',
        data_prefix='data/train',
        classes = 'data/classes.txt',
        ),
    val=dict(
        ann_file = 'data/val.txt',
        data_prefix='data/val',
        classes = 'data/classes.txt',
        ),
    test=dict(
        ann_file = 'data/val.txt',
        data_prefix='data/val',
        classes = 'data/classes.txt',
        ))


evaluation = dict(interval=1, metric='accuracy',metric_options={'topk': (1, )})
# metric_options={'topk': (1, )}

runner = dict(type='EpochBasedRunner', max_epochs=100)

load_from = 'pretrained/twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth'

model = dict(head=dict(
        num_classes=5,
        topk = (1,)
        ),
        train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=5, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=5, prob=0.5)
    ]))

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook' )
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='Your-project'))
    ])

# vis_backends = [dict(type='LocalvisBackend'),
#                 dict(type='WandbvisBackend') # can cancel wandb for debug
# ]
# visualizer = dict(type='clsVisualizer', vis_backends=vis_backends)