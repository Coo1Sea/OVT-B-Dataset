model = dict(
    type='OVTrack',
    freeze_detector=True,
    method='ovtrack-tao',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://detectron2/resnet50_caffe')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='OVTrackRoIHead',
        finetune_track=True,
        dynamic_rcnn_thre=False,
        prompt_path='saved_models/pretrained_models/detpro_prompt.pt',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared4Conv1FCCliPBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1203,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            ensemble=True,
            with_cls=False,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        track_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25,
                            version="unbiased"),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        embed=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))
    ),
    tracker=dict(
        type='OVTracker',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_frames=10,
        momentum_embed=0.8,
        momentum_obj_score=0.5,
        match_metric='cosine',
        match_with_cosine=True,
        contrastive_thr=0.5,
    ),
test_cfg = dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),

    rcnn=dict(
        score_thr=0.0001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300,
        mask_thr_binary=0.5),
)
)

dataset_type = 'TaoDataset'
data_root = 'data/tao/'

img_scale = (800, 1333)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='SeqMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='SeqRandomAffine'),
    dict(type='SeqYOLOXHSVRandomAug'),
    dict(type='SeqRandomFlip',share_params=False, flip_ratio=0.5),

    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=False,
        multiscale_mode='value',
        keep_ratio=True),

    dict(type='SeqNormalize', **img_norm_cfg),
    dict(
        type='SeqPad',
        size_divisor=32,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel
        ),
    dict(type='SeqFilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
        ref_prefix='ref'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadImageFromFile',
    #      file_client_args=dict(
    #          img_db_path='data/tao/tao_val_imgs.hdf5',
    #          backend='hdf5',
    #          type='tao')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

## datasets settings
dataset_type = 'TaoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=3,
    persistent_workers=True,
    train=dict(
            type='SeqMultiImageMixDataset',
            dataset=dict(
                type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(
                type=dataset_type,
                classes='data/lvis/annotations/lvis_classes_v1.txt',
                load_as_video=True,
                ann_file='data/tao/annotations/diff_track/ovtrack_lvis_generated_image_pairs_repeat_1.json',
                key_img_sampler=dict(interval=1),
                ref_img_sampler=dict(num_ref_imgs=1, scope=1, method='uniform', pesudo=True),
                pipeline=[
                    dict(
                        type='LoadMultiImagesFromFile',
                        file_client_args=dict(
                            img_db_path='data/tao/diff_track/ovtrack_lvis_generated_image_pairs_repeat_1.h5',
                            backend='hdf5',
                            type='lvis')),
                    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True),
                ])
            ),
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes='data/lvis/annotations/lvis_classes_v1.txt',
        ann_file='data/tao/annotations/validation_ours_v1.json',

        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes='data/lvis/annotations/lvis_classes_v1.txt',
        ann_file='data/tao/annotations/validation_ours_v1.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline)

)

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[3, 5])
total_epochs = 6
load_from = 'saved_models/pretrained_models/ovtrack_clip_distillation.pth'
evaluation = dict(metric=['track'], start=6, interval=1, resfile_path='/scratch/tmp/')

checkpoint_config = dict(interval=1, create_symlink=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
find_unused_parameters = True
