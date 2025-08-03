# # model settings
# model = dict(
#     type='Recognizer2D_ours',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet50',
#         depth=50,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead_ours',
#         num_classes=6,
#         in_channels=1408,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))


# body_head_model = dict(
#     type='Recognizer2D',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet34',
#         depth=34,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead',
#         num_classes=11,
#         in_channels=512,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))


# upper_limb_model = dict(
#     type='Recognizer2D',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet34',
#         depth=34,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead',
#         num_classes=13,
#         in_channels=512,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))


# lower_limb_model = dict(
#     type='Recognizer2D',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet34',
#         depth=34,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead',
#         num_classes=8,
#         in_channels=512,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))


# body_hand_model = dict(
#     type='Recognizer2D_ours',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet50',
#         depth=50,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead_ours',
#         num_classes=6,
#         in_channels=1408,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))


# head_hand_model = dict(
#     type='Recognizer2D_ours',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet50',
#         depth=50,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead_ours',
#         num_classes=10,
#         in_channels=1408,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))

# leg_hand_model = dict(
#     type='Recognizer2D_ours',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet50',
#         depth=50,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead_ours',
#         num_classes=4,
#         in_channels=1408,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))


# combined_model = dict(
#     type='MultiBranchModel',
#     backbone=dict(
#         type='ResNetTSM',
#         pretrained='torchvision://resnet50',
#         depth=50,
#         norm_eval=False,
#         shift_div=8),
#     cls_head=dict(
#         type='MANetHead_ours',
#         num_classes=4,
#         in_channels=1408,
#         spatial_type='avg',
#         consensus=dict(type='AvgConsensus', dim=1),
#         dropout_ratio=0.5,
#         init_std=0.001,
#         is_shift=True),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))

model = dict(
    type='MultiBranchModel',
    main_model=dict(
        type='Recognizer2D_ours',
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_backbone_for_expert_selector_mpii/best_top1_acc_epoch_13.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours_returns_cls_loss',
            num_classes=4,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    body_head_model=dict(
        type='Recognizer2D_ours',
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_backbone_for_experts_first_expert_mpii_passive/best_top1_acc_epoch_34.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=6,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    upper_limb_model=dict(
        type='Recognizer2D_ours',
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_backbone_for_experts_second_expert_mpii_passive/best_top1_acc_epoch_21.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=3,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    lower_limb_model=dict(
        type='Recognizer2D_ours',
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_backbone_for_experts_third_expert_mpii_passive/best_top1_acc_epoch_47.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=3,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    body_hand_model=dict(
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_backbone_for_experts_fourth_expert_mpii_passive/best_top1_acc_epoch_16.pth',
        type='Recognizer2D_ours',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=7,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    head_hand_model=dict(
        type='Recognizer2D_ours',
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_backbone_for_experts_fourth_expert_mpii_passive/best_top1_acc_epoch_16.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=7,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    leg_hand_model=dict(
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_backbone_for_experts_fourth_expert_mpii_passive/best_top1_acc_epoch_16.pth',
        type='Recognizer2D_ours',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=7,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    
    manet_52_model=dict(
        type='Recognizer2D',
        pretrained='/data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_for_mpII/best_top1_acc_epoch_39.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead',
            num_classes=19,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
)
