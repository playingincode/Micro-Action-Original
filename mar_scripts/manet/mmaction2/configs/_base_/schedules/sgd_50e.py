# optimizer
# optimizer = dict(
#     type='SGD',
#     lr=0.01,  # this lr is used for 8 gpus
#     momentum=0.9,
#     weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,  # base LR for main_model
        momentum=0.9,
        weight_decay=0.0001,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'module.body_head_model': dict(lr_mult=0.1),
            'module.upper_limb_model': dict(lr_mult=0.1),
            'module.lower_limb_model': dict(lr_mult=0.1),
            'module.body_hand_model': dict(lr_mult=0.1),
            'module.head_hand_model': dict(lr_mult=0.1),
            'module.leg_hand_model': dict(lr_mult=0.1),
        }
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
)
