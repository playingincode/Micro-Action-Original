import torch.nn as nn
from ..builder import MULTIMODAL
from ..builder import build_loss
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch.nn.functional as F
import torch
from ...core import top_k_accuracy
from collections import OrderedDict
import torch.distributed as dist

@MULTIMODAL.register_module()
class MultiBranchModel(nn.Module):
    def __init__(self,
                 main_model,
                 body_head_model,
                 upper_limb_model,
                 lower_limb_model,
                 body_hand_model,
                 head_hand_model,
                 leg_hand_model,
                 num_classes=52,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_emb=dict(type='MseLoss'),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=(1, 5)):
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
        #  super().__init__()
        self.num_classes = num_classes
        self.in_channels = 1408
        self.loss_cls = build_loss(loss_cls)
        self.loss_emb = build_loss(loss_emb)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    def forward_test(self,imgs,label,emb, videomae_features,**kwargs):
        # imgs = data_batch['imgs']
        # label = data_batch['label']
        # emb=data_batch['emb']
        # videomae_features=data_batch['videomae_features']
        
        out_main = self.main_model(imgs, label,emb, videomae_features,**kwargs)
        # print("Out main",out_main)
        # print(out_main)
        out_body_head = self.body_head_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_body_head)
        out_upper_limb = self.upper_limb_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_upper_limb)
        out_lower_limb = self.lower_limb_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_lower_limb)
        out_body_hand = self.body_hand_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_body_hand)
        out_head_hand= self.head_hand_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_head_hand)
        out_leg_hand = self.leg_hand_model(imgs, label,emb, videomae_features,**kwargs)
        
        
        # print("emb_score_main:", emb_score_main.shape)
        # print("emb_score_body_head:", emb_score_body_head.shape)
        # print("emb_score_upper_limb:", emb_score_upper_limb.shape)
        # print("emb_score_lower_limb:", emb_score_lower_limb.shape)
        # print("emb_score_body_hand:", emb_score_body_hand.shape)
        # print("emb_score_head_hand:", emb_score_head_hand.shape)
        # print("emb_score_leg_hand:", emb_score_leg_hand.shape)
        
        # emb_score_0 = emb_score_body_head
        # emb_score_1 = emb_score_upper_limb
        # emb_score_2 = emb_score_lower_limb
        # emb_score_3 = emb_score_body_hand
        # emb_score_4 = emb_score_head_hand
        # emb_score_5 = emb_score_leg_hand
        # print(out_leg_hand)
        # emb_scores = torch.stack([
        #     emb_score_0,
        #     emb_score_1,
        #     emb_score_2,
        #     emb_score_3,
        #     emb_score_4,
        #     emb_score_5
        # ], dim=1)
        # weights = F.softmax(out_main, dim=1) 
        gate_weights = torch.softmax(out_main, dim=1)
        device = out_main.device
        num_classes = 52
        # print("Weights",weights)
        expert_class_indices = [
            list(range(0, 11)),    # Expert 0
            list(range(11, 24)),   # Expert 1
            list(range(24, 32)),   # Expert 2
            list(range(32, 38)),   # Expert 3
            list(range(38, 48)),   # Expert 4
            list(range(48, 52)),   # Expert 5
        ]
        # print(self.expand_to_52(out_body_head, expert_class_indices[0]))
        expert_outputs = [
            self.expand_to_52(out_body_head, expert_class_indices[0]),
            self.expand_to_52(out_upper_limb, expert_class_indices[1]),
            self.expand_to_52(out_lower_limb, expert_class_indices[2]),
            self.expand_to_52(out_body_hand, expert_class_indices[3]),
            self.expand_to_52(out_head_hand, expert_class_indices[4]),
            self.expand_to_52(out_leg_hand, expert_class_indices[5]),
        ]  # each is [B, 52], 6, D]
        # weights = weights.unsqueeze(-1)
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
        # weights = weights.unsqueeze(-1)
        weighted_expert_outputs = gate_weights.unsqueeze(-1) * expert_outputs_stacked
        final_logits = weighted_expert_outputs.sum(dim=1)
        # final_emb_score = torch.sum(gate_weights.unsqueeze(-1) * emb_scores, dim=1)
        # gt_labels = label.squeeze()
        # loss=dict()
        # loss_cls = self.loss(final_logits, final_emb_score,gt_labels,emb, **kwargs)
        # loss.update(loss_cls)
        # loss, log_vars = self._parse_losses(loss)
        
        # outputs = dict(
        #     loss=loss,
        #     log_vars=log_vars,
        #     num_samples=len(next(iter(data_batch.values()))))

        return final_logits
    
    def forward(self, imgs, label,emb,videomae_features, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        # return_loss=False
        # print("Return loss inside unified model",return_loss)
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            return self.forward_train_with_logits(imgs, label,emb,videomae_features, **kwargs)
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
            
        kwargs['return_loss'] = return_loss
        return self.forward_test(imgs,label,emb, videomae_features,**kwargs).cpu().numpy()
 
    def expand_to_52(self,logits, class_indices):
        num_classes = 52
        full = torch.zeros((logits.size(0), num_classes), device=logits.device)
        full[:, class_indices] = logits
        return full


    def loss(self, cls_score, emb_score,labels,embs_la, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        loss_embd=self.loss_emb(emb_score,embs_la,labels)*50
        loss_cls+=loss_embd
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
    
    def _parse_losses(self,losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
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
        
        out_main,emb_score_main = self.main_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_main)
        out_body_head,emb_score_body_head = self.body_head_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_body_head)
        out_upper_limb,emb_score_upper_limb = self.upper_limb_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_upper_limb)
        out_lower_limb,emb_score_lower_limb = self.lower_limb_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_lower_limb)
        out_body_hand,emb_score_body_hand = self.body_hand_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_body_hand)
        out_head_hand,emb_score_head_hand= self.head_hand_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_head_hand)
        out_leg_hand,emb_score_leg_hand = self.leg_hand_model(imgs, label,emb, videomae_features,**kwargs)
        
        
        # print("emb_score_main:", emb_score_main.shape)
        # print("emb_score_body_head:", emb_score_body_head.shape)
        # print("emb_score_upper_limb:", emb_score_upper_limb.shape)
        # print("emb_score_lower_limb:", emb_score_lower_limb.shape)
        # print("emb_score_body_hand:", emb_score_body_hand.shape)
        # print("emb_score_head_hand:", emb_score_head_hand.shape)
        # print("emb_score_leg_hand:", emb_score_leg_hand.shape)
        
        emb_score_0 = emb_score_body_head
        emb_score_1 = emb_score_upper_limb
        emb_score_2 = emb_score_lower_limb
        emb_score_3 = emb_score_body_hand
        emb_score_4 = emb_score_head_hand
        emb_score_5 = emb_score_leg_hand
        # print(out_leg_hand)
        emb_scores = torch.stack([
            emb_score_0,
            emb_score_1,
            emb_score_2,
            emb_score_3,
            emb_score_4,
            emb_score_5
        ], dim=1)
        # weights = F.softmax(out_main, dim=1) 
        gate_weights = torch.softmax(out_main, dim=1)
        device = out_main.device
        num_classes = 52
        # print("Weights",weights)
        expert_class_indices = [
            list(range(0, 11)),    # Expert 0
            list(range(11, 24)),   # Expert 1
            list(range(24, 32)),   # Expert 2
            list(range(32, 38)),   # Expert 3
            list(range(38, 48)),   # Expert 4
            list(range(48, 52)),   # Expert 5
        ]
        # print(self.expand_to_52(out_body_head, expert_class_indices[0]))
        expert_outputs = [
            self.expand_to_52(out_body_head, expert_class_indices[0]),
            self.expand_to_52(out_upper_limb, expert_class_indices[1]),
            self.expand_to_52(out_lower_limb, expert_class_indices[2]),
            self.expand_to_52(out_body_hand, expert_class_indices[3]),
            self.expand_to_52(out_head_hand, expert_class_indices[4]),
            self.expand_to_52(out_leg_hand, expert_class_indices[5]),
        ]  # each is [B, 52], 6, D]
        # weights = weights.unsqueeze(-1)
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
        # weights = weights.unsqueeze(-1)
        weighted_expert_outputs = gate_weights.unsqueeze(-1) * expert_outputs_stacked
        final_logits = weighted_expert_outputs.sum(dim=1)
        final_emb_score = torch.sum(gate_weights.unsqueeze(-1) * emb_scores, dim=1)
        loss=dict()
        gt_labels = label.squeeze()
        loss_cls = self.loss(final_logits, final_emb_score,gt_labels,emb, **kwargs)
        loss.update(loss_cls)
        loss, log_vars = self._parse_losses(loss)
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
        # loss
        
        
        # print(final_logits)
        
        
        # losses = dict()
        # if labels.shape == torch.Size([]):
        #     labels = labels.unsqueeze(0)
        # elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
        #         and cls_score.size()[0] == 1:
        #     # Fix a bug when training with soft labels and batch size is 1.
        #     # When using soft labels, `labels` and `cls_socre` share the same
        #     # shape.
        #     labels = labels.unsqueeze(0)

        # if not self.multi_class and cls_score.size() != labels.size():
        #     top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
        #                                labels.detach().cpu().numpy(),
        #                                self.topk)
        #     for k, a in zip(self.topk, top_k_acc):
        #         losses[f'top{k}_acc'] = torch.tensor(
        #             a, device=cls_score.device)

        # elif self.multi_class and self.label_smooth_eps != 0:
        #     labels = ((1 - self.label_smooth_eps) * labels +
        #               self.label_smooth_eps / self.num_classes)

        # loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_embd=self.loss_emb(emb_score,embs_la,labels)*50
        # loss_cls+=loss_embd
        # # loss_cls may be dictionary or single tensor
        # if isinstance(loss_cls, dict):
        #     losses.update(loss_cls)
        # else:
        #     losses['loss_cls'] = loss_cls

        
        
        


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
        
        out_main,emb_score_main = self.main_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_main)
        out_body_head,emb_score_body_head = self.body_head_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_body_head)
        out_upper_limb,emb_score_upper_limb = self.upper_limb_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_upper_limb)
        out_lower_limb,emb_score_lower_limb = self.lower_limb_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_lower_limb)
        out_body_hand,emb_score_body_hand = self.body_hand_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_body_hand)
        out_head_hand,emb_score_head_hand= self.head_hand_model(imgs, label,emb, videomae_features,**kwargs)
        # print(out_head_hand)
        out_leg_hand,emb_score_leg_hand = self.leg_hand_model(imgs, label,emb, videomae_features,**kwargs)
        
        
        # print("emb_score_main:", emb_score_main.shape)
        # print("emb_score_body_head:", emb_score_body_head.shape)
        # print("emb_score_upper_limb:", emb_score_upper_limb.shape)
        # print("emb_score_lower_limb:", emb_score_lower_limb.shape)
        # print("emb_score_body_hand:", emb_score_body_hand.shape)
        # print("emb_score_head_hand:", emb_score_head_hand.shape)
        # print("emb_score_leg_hand:", emb_score_leg_hand.shape)
        
        emb_score_0 = emb_score_body_head
        emb_score_1 = emb_score_upper_limb
        emb_score_2 = emb_score_lower_limb
        emb_score_3 = emb_score_body_hand
        emb_score_4 = emb_score_head_hand
        emb_score_5 = emb_score_leg_hand
        # print(out_leg_hand)
        emb_scores = torch.stack([
            emb_score_0,
            emb_score_1,
            emb_score_2,
            emb_score_3,
            emb_score_4,
            emb_score_5
        ], dim=1)
        # weights = F.softmax(out_main, dim=1) 
        gate_weights = torch.softmax(out_main, dim=1)
        device = out_main.device
        num_classes = 52
        # print("Weights",weights)
        expert_class_indices = [
            list(range(0, 11)),    # Expert 0
            list(range(11, 24)),   # Expert 1
            list(range(24, 32)),   # Expert 2
            list(range(32, 38)),   # Expert 3
            list(range(38, 48)),   # Expert 4
            list(range(48, 52)),   # Expert 5
        ]
        # print(self.expand_to_52(out_body_head, expert_class_indices[0]))
        expert_outputs = [
            self.expand_to_52(out_body_head, expert_class_indices[0]),
            self.expand_to_52(out_upper_limb, expert_class_indices[1]),
            self.expand_to_52(out_lower_limb, expert_class_indices[2]),
            self.expand_to_52(out_body_hand, expert_class_indices[3]),
            self.expand_to_52(out_head_hand, expert_class_indices[4]),
            self.expand_to_52(out_leg_hand, expert_class_indices[5]),
        ]  # each is [B, 52], 6, D]
        # weights = weights.unsqueeze(-1)
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
        # weights = weights.unsqueeze(-1)
        weighted_expert_outputs = gate_weights.unsqueeze(-1) * expert_outputs_stacked
        final_logits = weighted_expert_outputs.sum(dim=1)
        final_emb_score = torch.sum(gate_weights.unsqueeze(-1) * emb_scores, dim=1)
        gt_labels = label.squeeze()
        loss=dict()
        loss_cls = self.loss(final_logits, final_emb_score,gt_labels,emb, **kwargs)
        loss.update(loss_cls)
        loss, log_vars = self._parse_losses(loss)
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs