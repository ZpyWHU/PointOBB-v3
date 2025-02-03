_base_ = [
    '../../configs/_base_/default_runtime.py'
]

angle_version = 'le90'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add
debug = False

num_stages = 2
model = dict(
    type='PointOBB',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4, 
        norm_cfg=norm_cfg
    ),
    
    loss_diff_view=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),  # SSC loss
    crop_size = (800, 800),
    construct_view = True,   # rot/flp view
    construct_resize = True, # resized view
    weight_mode = 'dynamic_weight',
    scale_classes = [9, 14, 19],
    scale_classes2 = [5, 15],
    
    roi_head=dict(
        type='PointOBBHead',
        num_stages=num_stages,
        top_k=7,
        with_atten=False,

        loss_symmetry_ss=dict(
            type='SmoothL1Loss', loss_weight=0.5, beta=0.1),
        angle_coder=dict(
                    type='PSCCoder',
                    angle_version='le90',
                    dual_freq=False,
                    num_step=3,
                    thr_mod=0),
        angle_version = 'le90',
        rotation_agnostic_classes=[2, 5, 9, 14, 15, 19],
        agnostic_resize_classes = [5, 13, 15, 18],
        use_angle_loss = False,
        add_angle_pred_begin = False,
        not_use_rot_mil = False, 
        detach_angle_head = False,
        stacked_convs = 2,
        use_ssff = True,

        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),

        bbox_head=dict(
            type='Shared2FCInstanceMILHead',
            num_stages=num_stages,
            with_loss_pseudo=False,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            num_ref_fcs=0,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_type='MIL',
            loss_mil1=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='binary_cross_entropy'), 
            loss_mil2=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='gfocal_loss'),),
    ),

    bbox_pred_head=dict(
        type='PointOBB_FCOS_Head',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        h_bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=None,
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ),


    # model training and testing settings
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 24, 32, 48, 64, 72, 80, 96],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            shake_ratio=None,
            cut_mode='symmetry', 
            gen_num_neg=0),
        
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            shake_ratio=[0.1],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            iou_thr=0.3,
            gen_num_neg=500,
        ),
        rcnn=None,
        iter_count = 0,
        burn_in_steps1 = 16000, 
        burn_in_steps2 = 22000,
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=None,
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# dataset settings
dataset_type = 'DIORPointDataset'
angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5,version=angle_version) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    # dict(
    #     type='RandomFlip',
    #     flip_ratio=[0.25, 0.25, 0.25],
    #     direction=['horizontal', 'vertical', 'diagonal'],
    #     version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_true_bboxes']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(800, 800) , keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])

]
data_root_trainval = '/datasets/DIOR/'
data_root_test = '/datasets/DIOR/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,  
    shuffle=False if debug else None,
    train=dict(
        type=dataset_type,
        version=angle_version,
        ann_file = data_root_trainval + "Annotations/trainval_rbox_pt_P2Bfmt.json",
        img_prefix = data_root_trainval + 'JPEGImages-trainval/',
        pipeline=train_pipeline,
        filter_empty_gt=True
    ),
    val=dict(
        samples_per_gpu=8,
        type=dataset_type,
        ann_file = data_root_trainval + "Annotations/trainval_rbox_pt_P2Bfmt.json",
        img_prefix = data_root_trainval + 'JPEGImages-trainval/', 
        pipeline=test_pipeline,
        test_mode=False,
    ),
    # test=dict(
    #     type=dataset_type,
    #     img_prefix=data_root_test + 'JPEGImages-testfordebug/',
    #     ann_file=data_root_test + "Annotations/testfordebug_rbox_pt_P2Bfmt.json",
    #     pipeline=test_pipeline))
    test=dict(
        type=dataset_type,
        ann_file=data_root_test + "Annotations/test_rbox_pt_P2Bfmt.json",
        img_prefix=data_root_test + 'JPEGImages-test/',
        pipeline=test_pipeline))

check = dict(stop_while_nan=False)

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
training_time = 2
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8*training_time, 11*training_time])

runner = dict(type='EpochBasedRunner', max_epochs=12*training_time)

checkpoint_config = dict(interval=1)

