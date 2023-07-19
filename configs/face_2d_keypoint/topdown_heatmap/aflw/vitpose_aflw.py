_base_ = [
    '../../../body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py'
]


# hooks
default_hooks = dict(
    checkpoint=dict(save_best='aflw/AP', rule='greater', max_keep_ckpts=1))


# model settings
model = dict(
    _delete_=True,
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
    ),
    neck=dict(type='FeatureMapProcessor', scale_factor=4.0, apply_relu=True),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=19,
        deconv_out_channels=[],
        deconv_kernel_sizes=[],
        final_layer=dict(kernel_size=3, padding=1),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))
data_root = 'data/aflw/'
dataset_type = 'AFLWDataset'
data_mode = 'topdown'


# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='annotations/face_landmarks_aflw_train.json',
        data_prefix=dict(img='images/'),
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        ann_file='annotations/face_landmarks_aflw_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='AFLWMetric',
    ann_file=data_root + 'annotations/face_landmarks_aflw_test.json')
test_evaluator = val_evaluator

