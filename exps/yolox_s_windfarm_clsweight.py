#!/usr/bin/env python3
# Measure (B): class-weighted classification loss (emphasise cable tower). Only the cls loss differs
# from the baseline (exps/yolox_s_windfarm.py); schedule/data/aug are inherited unchanged.
#
# Hypothesis: cable tower is ~35x rarer than turbine (441 vs 15534 train instances), so the cls loss
# is dominated by turbine anchors. Up-weighting the anchors assigned to a cable tower should raise
# its recall/AP.
#
# Env vars (in addition to the baseline's YOLOX_DATA_DIR / EVAL_SPLIT):
#   CLS_WEIGHTS  "w0,w1" per-class weights, index 0 = cable tower, 1 = turbine (default "4.0,1.0")
#   CLS_PRIOR    "n0,n1" class counts used only for normalisation (default "441,15534" = train)
#
# Scheme: each foreground anchor gets the weight of its matched GT class, applied to ALL class logits
# of that anchor (target = one-hot * IoU: the positive and the 0-target of the other class).
# For the cable-tower logit this up-weights positives (tower anchors) while its negatives (turbine
# anchors) keep weight ~1, i.e. it shifts the pos/neg balance. (A per-logit-column weight would scale
# pos and neg of that logit equally and not change the balance.) Non-fg anchors are not in YOLOX's
# cls loss at all. Obj and reg losses are untouched.
# Normalisation: weights are divided by sum_c(w_c * p_c), p_c = CLS_PRIOR frequency, so the expected
# weight per fg anchor is 1 and the cls-loss scale stays ~ baseline (with "4,1": tower 3.69,
# turbine 0.92). Static, so no per-batch fluctuation. CLS_WEIGHTS="1,1" == baseline loss exactly.
#
# get_losses below is a copy of upstream YOLOXHead.get_losses; changed lines are marked [CLSW].
#
# Train:  python tools/train.py -f exps/yolox_s_windfarm_clsweight.py -d 1 -b 16 --fp16 -o -c yolox_s.pth
import os

import torch
import torch.nn.functional as F
from loguru import logger

from yolox.models import YOLOXHead

from yolox_s_windfarm import Exp as BaseExp  # same dir (exps/) is on sys.path, like the baseline


def _parse_floats(env_name, default, n):
    vals = [float(v) for v in os.environ.get(env_name, default).split(",")]
    if len(vals) != n or any(v <= 0 for v in vals):
        raise ValueError(f"{env_name} must be {n} positive comma-separated numbers, got {vals}")
    return vals


class ClsWeightedHead(YOLOXHead):
    def __init__(self, *args, cls_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        w = torch.tensor(cls_weights, dtype=torch.float32)
        # non-persistent: keeps state_dict identical to the baseline's (checkpoints stay compatible)
        self.register_buffer("cls_weights", w, persistent=False)

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        cls_weights = []  # [CLSW] per-fg-anchor weight (by matched gt class)
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                cls_weight = outputs.new_zeros((0, 1))  # [CLSW]
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                cls_weight = self.cls_weights[gt_matched_classes.to(torch.int64)].unsqueeze(-1)  # [CLSW]
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            cls_weights.append(cls_weight)  # [CLSW]
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        cls_weights = torch.cat(cls_weights, 0).to(cls_preds.dtype)  # [CLSW]
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
            * cls_weights  # [CLSW] scales all class logits of each fg anchor
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # baseline derives exp_name from ITS __file__; override so outputs do not collide
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        w = _parse_floats("CLS_WEIGHTS", "4.0,1.0", self.num_classes)
        prior = _parse_floats("CLS_PRIOR", "441,15534", self.num_classes)
        norm = sum(wi * ci for wi, ci in zip(w, prior)) / sum(prior)
        self.cls_weights = [wi / norm for wi in w]

    def get_model(self):
        # copy of upstream Exp.get_model with the head swapped for ClsWeightedHead
        import torch.nn as nn
        from yolox.models import YOLOX, YOLOPAFPN

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = ClsWeightedHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act,
                cls_weights=self.cls_weights,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
