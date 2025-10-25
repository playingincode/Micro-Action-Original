optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',  # keeps MMACTION2 paramwise handling
    lr=0.01,  # base LR (used for main_model)
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        # lower LR for expert submodels
        custom_keys={
            'module.body_head_model': dict(lr_mult=0.1),
            'module.upper_limb_model': dict(lr_mult=0.1),
            'module.lower_limb_model': dict(lr_mult=0.1),
            'module.body_hand_model': dict(lr_mult=0.1),
            'module.head_hand_model': dict(lr_mult=0.1),
            'module.leg_hand_model': dict(lr_mult=0.1),
            # optional: keep main_model at base LR
            'module.main_model': dict(lr_mult=1.0),
        },
        # If your TSM constructor supports fc_lr5, you can keep it for head layers
        fc_lr5=True
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[30, 60])
total_epochs = 500