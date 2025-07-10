import torch.nn as nn
from ..builder import MULTIMODAL
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
@MULTIMODAL.register_module()

class MultiBranchModel(nn.Module):
    def __init__(self,
                 main_model,
                 body_head_model,
                 upper_limb_model,
                 lower_limb_model,
                 body_hand_model,
                 head_hand_model,
                 leg_hand_model):
        super().__init__()

        # Build submodels from config dictionaries
        self.main_model = build_model(main_model)
        self.body_head_model = build_model(body_head_model)
        self.upper_limb_model = build_model(upper_limb_model)
        self.lower_limb_model = build_model(lower_limb_model)
        self.body_hand_model = build_model(body_hand_model)
        self.head_hand_model = build_model(head_hand_model)
        self.leg_hand_model = build_model(leg_hand_model)
        load_checkpoint(self.main_model, main_model.pretrained, map_location='cpu')
        load_checkpoint(self.body_head_model, body_head_model.pretrained, map_location='cpu')
        load_checkpoint(self.upper_limb_model, upper_limb_model.pretrained, map_location='cpu')
        load_checkpoint(self.lower_limb_model, lower_limb_model.pretrained, map_location='cpu')
        load_checkpoint(self.body_hand_model, body_hand_model.pretrained, map_location='cpu')
        load_checkpoint(self.head_hand_model, head_hand_model.pretrained, map_location='cpu')
        load_checkpoint(self.leg_hand_model, leg_hand_model.pretrained, map_location='cpu')

    def forward(self, inputs, labels=None, **kwargs):
        # Forward each model
        out_main = self.main_model(inputs, labels=labels, **kwargs)
        out_body_head = self.body_head_model(inputs, labels=labels, **kwargs)
        out_upper_limb = self.upper_limb_model(inputs, labels=labels, **kwargs)
        out_lower_limb = self.lower_limb_model(inputs, labels=labels, **kwargs)
        out_body_hand = self.body_hand_model(inputs, labels=labels, **kwargs)
        out_head_hand = self.head_hand_model(inputs, labels=labels, **kwargs)
        out_leg_hand = self.leg_hand_model(inputs, labels=labels, **kwargs)

        # Assume each output is a dict with 'loss' key
        total_loss = (
            out_main['loss'] +
            out_body_head['loss'] +
            out_upper_limb['loss'] +
            out_lower_limb['loss'] +
            out_body_hand['loss'] +
            out_head_hand['loss'] +
            out_leg_hand['loss']
        )

        return dict(
            loss=total_loss,
            losses={
                'main': out_main,
                'body_head': out_body_head,
                'upper_limb': out_upper_limb,
                'lower_limb': out_lower_limb,
                'body_hand': out_body_hand,
                'head_hand': out_head_hand,
                'leg_hand': out_leg_hand,
            }
        )


    def train_step(self, data_batch, optimizer, **kwargs):

        """The iteration step during training.
        训练期间的迭代步骤。
        此方法定义了训练期间的迭代步骤
        反向传播和优化器更新，在优化器中完成
        钩请注意，在一些复杂的情况或模型中，整个过程
        包括反向传播和优化器更新也在中定义
        例如GAN。

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        
        imgs = data_batch['imgs']
        label = data_batch['label']
        emb=data_batch['emb']
        videomae_features=data_batch['videomae_features']
        
        out_main = self.main_model(imgs, label,emb, videomae_features,**kwargs)
        print(out_main)
        out_body_head = self.body_head_model(imgs, label,emb, videomae_features,**kwargs)
        print(out_body_head)
        out_upper_limb = self.upper_limb_model(imgs, label,emb, videomae_features,**kwargs)
        print(out_upper_limb)
        out_lower_limb = self.lower_limb_model(imgs, label,emb, videomae_features,**kwargs)
        print(out_lower_limb)
        out_body_hand = self.body_hand_model(imgs, label,emb, videomae_features,**kwargs)
        print(out_body_hand)
        out_head_hand = self.head_hand_model(imgs, label,emb, videomae_features,**kwargs)
        print(out_head_hand)
        out_leg_hand = self.leg_hand_model(imgs, label,emb, videomae_features,**kwargs)
        print(out_leg_hand)


        return 0

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        # """
        imgs = data_batch['imgs']
        label = data_batch['label']
        emb=data_batch['emb']
        videomae_features=data_batch['videomae_features']

        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = data_batch[item]

        losses = self(imgs, label,emb, videomae_features,return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs