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
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/experts/expert_selector_MPII_running_changed_epochs/best_top1_acc_epoch_14.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours_returns_cls_loss',
            num_classes=4,
            in_channels=1408,
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
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/experts/first_expert_MPII_running_changed_epochs/best_top1_acc_epoch_25.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=6,
            in_channels=1408,
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
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/experts/second_expert_running_for_mpii_again/best_top1_acc_epoch_6.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=3,
            in_channels=1408,
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
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/experts/third_expert_MPII_running_changed_epochs/best_top1_acc_epoch_2.pth',
        backbone=dict(
            type='ResNetTSM',
            pretrained='torchvision://resnet50',
            depth=50,
            norm_eval=False,
            shift_div=8),
        cls_head=dict(
            type='MANetHead_ours',
            num_classes=3,
            in_channels=1408,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    body_hand_model=dict(
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/experts/fourth_expert_MPII_running_changed_epochs/best_top1_acc_epoch_8.pth',
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
            in_channels=1408,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    head_hand_model=dict(
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/experts/fourth_expert_MPII_running_changed_epochs/best_top1_acc_epoch_8.pth',
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
            in_channels=1408,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.5,
            init_std=0.001,
            is_shift=True),
        train_cfg=None,
        test_cfg=dict(average_clips='prob')
    ),
    leg_hand_model=dict(
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/experts/fourth_expert_MPII_running_changed_epochs/best_top1_acc_epoch_8.pth',
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
            in_channels=1408,
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
        pretrained='/srv/storage/stars@storage3.sophia.grid5000.fr/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_for_mpII/best_top1_acc_epoch_39.pth',
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



checkpoint_six_classes="/data/stars/user/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_prova_sgp_six_classes_our_model/best_top1_acc_epoch_32.pth"
body_head_ckpt="/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/MICRO_6CLASSES/Micro-Action-Original/mar_scripts/manet/mmaction2/work_dirs/manet_resnet34_body_head/best_top1_acc_epoch_39.pth"
upper_limb_ckpt="/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/MICRO_6CLASSES/Micro-Action-Original/mar_scripts/manet/mmaction2/work_dirs/manet_resnet34_upper_limb/best_top1_acc_epoch_58.pth"
lower_limb_ckpt="/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/MICRO_6CLASSES/Micro-Action-Original/mar_scripts/manet/mmaction2/work_dirs/manet_resnet34_lower_limb/best_top1_acc_epoch_64.pth"
body_hand_ckpt="/data/stars/user/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_prova_sgp_body_hand_passive/best_top1_acc_epoch_11.pth"
head_hand_ckpt="/data/stars/user/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_prova_sgp_head_hand_passive/best_top1_acc_epoch_7.pth"
leg_hand_ckpt="/data/stars/user/npoddar/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/manet_prova_sgp_leg_hand/best_top1_acc_epoch_33.pth"