import copy

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ..builder import build_head
import copy
from torch.nn import functional as F
from ..builder import HEADS, build_loss
import math
from typing import Tuple, Union
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision import transforms

from .P2BNet import gen_proposals_from_cfg
from .utils import resize_proposal, resize_single_proposal, flip_tensor, hboxlist2cxcywha \
                   ,merge_batch_list, split_batch_list, box_iou_rotated, obb2poly_np
import cv2
import os
from mmdet.core.bbox.transforms import rbbox2result

from third_parties.ted.ted import TED

custom_colors = [
(255, 0, 0), # Blue
(0, 255, 0), # Green
(0, 255, 255), # Yellow
(0, 128, 255), # Light Green
(255, 0, 128), # Light Blue
(128, 0, 0), # Dark Blue
(0, 128, 0), # Dark Green
]


def resize_image(inputs, resize_ratio=0.5):
    down_inputs = F.interpolate(inputs, 
                                scale_factor=resize_ratio, 
                                mode='nearest')
    
    return down_inputs

def scale_bbox(bbox_results, class_index):
    for image_bboxes in bbox_results:
        class_bboxes = image_bboxes[class_index]
        if class_bboxes.shape[0] > 0:
            class_bboxes[:, 2] *= 0.65
            class_bboxes[:, 3] *= 0.65
            image_bboxes[class_index] = class_bboxes
    return bbox_results

def scale_bbox2(bbox_results, class_index):
    for image_bboxes in bbox_results:
        class_bboxes = image_bboxes[class_index]
        if class_bboxes.shape[0] > 0:
            class_bboxes[:, 2] *= 1.2
            class_bboxes[:, 3] *= 1.2
            image_bboxes[class_index] = class_bboxes
    return bbox_results

# my：P2B中PBR的正样本采样过程  输入：上一阶段生成的伪框  配置  图像元数据  阶段  输出：对伪框经过增强（调整尺寸和抖动）后的样本框以及有效位
def fine_rotate_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta, stage):
    gen_mode = fine_proposal_cfg['gen_proposal_mode']
    # cut_mode = fine_proposal_cfg['cut_mode']
    cut_mode = None
    if isinstance(fine_proposal_cfg['base_ratios'], tuple):
        base_ratios = fine_proposal_cfg['base_ratios'][stage - 1]
        shake_ratio = fine_proposal_cfg['shake_ratio'][stage - 1]
    else:
        base_ratios = fine_proposal_cfg['base_ratios']
        shake_ratio = fine_proposal_cfg['shake_ratio']
    if gen_mode == 'fix_gen':
        proposal_list = []
        proposals_valid_list = []
        for i in range(len(img_meta)):
            pps = []
            base_boxes = pseudo_boxes[i]
            for ratio_w in base_ratios:
                for ratio_h in base_ratios:
                    base_boxes_ = base_boxes.clone()
                    base_boxes_[:, 2] *= ratio_w
                    base_boxes_[:, 3] *= ratio_h
                    pps.append(base_boxes_.unsqueeze(1))
            pps_old = torch.cat(pps, dim=1)
            if shake_ratio is not None:
                pps_new = []
                pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 5))
                for ratio in shake_ratio:
                    pps = pps_old.clone()
                    pps_center = pps[:, :, :2]
                    pps_wh = pps[:, :, 2:4]
                    pps_angle = pps[:, :, 4].unsqueeze(2)
                    pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
                    pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
                    pps_y_t = pps_center[:, :, 1] - ratio * pps_wh[:, :, 1]
                    pps_y_d = pps_center[:, :, 1] + ratio * pps_wh[:, :, 1]
                    pps_center_l = torch.stack([pps_x_l, pps_center[:, :, 1]], dim=-1)
                    pps_center_r = torch.stack([pps_x_r, pps_center[:, :, 1]], dim=-1)
                    pps_center_t = torch.stack([pps_center[:, :, 0], pps_y_t], dim=-1)
                    pps_center_d = torch.stack([pps_center[:, :, 0], pps_y_d], dim=-1)
                    pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)
                    pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
                    pps_angle = pps_angle.unsqueeze(2).expand((pps_center.size()[0], pps_center.size()[1], pps_center.size()[2], 1))
                    pps = torch.cat([pps_center, pps_wh, pps_angle], dim=-1)
                    pps = pps.reshape(pps.shape[0], -1, 5)
                    pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 5))
                pps_new = torch.cat(pps_new, dim=2)
            else:
                pps_new = pps_old
            h, w, _ = img_meta[i]['img_shape']
            if cut_mode is 'clamp':
                pps_new[..., 0:4:2] = torch.clamp(pps_new[..., 0:4:2], 0, w)
                pps_new[..., 1:4:2] = torch.clamp(pps_new[..., 1:4:2], 0, h)
                proposals_valid_list.append(pps_new.new_full(
                    (*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1))
            else:
                rot_theta = base_boxes[:,-1].mean()
                img_xywh = pps_new.new_tensor([w/2, h/2, w, h, rot_theta])  # (cx,cy,w,h,theta)
                iof_in_img = box_iou_rotated(pps_new.reshape(-1, 5), img_xywh.unsqueeze(0), mode='iof')
                proposals_valid = iof_in_img > 0.8
            proposals_valid_list.append(proposals_valid)
            proposal_list.append(pps_new.reshape(-1, 5))

    return proposal_list, proposals_valid_list

# my：P2B中PBR的负样本采样过程   输入：点标签  配置  正样本  图像数据   输出：负样本框  负样本框权重
def gen_rotate_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    num_neg_gen = proposal_cfg['gen_num_neg']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    device = gt_points[0].device
    for i in range(len(gt_points)):
        pos_box = aug_generate_proposals[i]
        h, w, _ = img_meta[i]['img_shape']
        x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
        y1 = -0.2 * h + torch.rand(num_neg_gen) * (1.2 * h)
        x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
        y2 = y1 + torch.rand(num_neg_gen) * (1.2 * h - y1)
        neg_theta = torch.ones_like(x1)*(pos_box[:,-1].mean().cpu())
        neg_bboxes = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2,
                                   x2 - x1, y2 - y1, neg_theta], dim=1).to(device)
        iou = box_iou_rotated(neg_bboxes, pos_box)
        neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])

        neg_proposal_list.append(neg_bboxes)
        neg_weight_list.append(neg_weight)
    return neg_proposal_list, neg_weight_list

def resize_rotate_proposal(img_metas,
                           batch_gt_bboxes, 
                           batch_proposals, 
                           gt_true_bboxes,
                           gt_bboxes_ignore,
                           ratio = 0.5):
    '''
    batch_gt_bboxes_all: [batch_size, num_proposals, 5] [cx,cy,w,h,a]
    batch_proposals_all: [batch_size, num_proposals, 5] [cx,cy,w,h,a]
    '''
    
    img_meta_out = copy.deepcopy(img_metas)
    batch_gt_bboxes_out = []
    batch_proposals_out =[]
    gt_true_bboxes_out = []
    gt_bboxes_ignore_out = []
    for i in range(len(img_metas)):
        h, w, c = img_metas[i]['img_shape']
        img_meta_out[i]['img_shape'] = (math.ceil(h * ratio), math.ceil(w * ratio), c)
        img_meta_out[i]['pad_shape'] = (math.ceil(h * ratio), math.ceil(w * ratio), c)
        tmp_gt_bboxes = batch_gt_bboxes[i].clone()
        tmp_gt_bboxes[:,:4] = tmp_gt_bboxes[:,:4] * ratio
        batch_gt_bboxes_out.append(tmp_gt_bboxes)

        tmp_proposal = batch_proposals[i].clone()
        tmp_proposal[:,:4] = tmp_proposal[:,:4] * ratio
        batch_proposals_out.append(tmp_proposal)

        tmp_gt_true_bbox = gt_true_bboxes[i].clone()
        tmp_gt_true_bbox[:,:4] = tmp_gt_true_bbox[:,:4] * ratio
        gt_true_bboxes_out.append(tmp_gt_true_bbox)
        
        tmp_gt_bboxes_ignore = gt_bboxes_ignore[i].clone()
        if gt_bboxes_ignore[i].size(0) != 0:
            tmp_gt_bboxes_ignore[:,:,:4] = tmp_gt_bboxes_ignore[:,:4] * ratio
        gt_bboxes_ignore_out.append(tmp_gt_bboxes_ignore)

    return img_meta_out, batch_gt_bboxes_out, batch_proposals_out, gt_true_bboxes_out, gt_bboxes_ignore_out

@DETECTORS.register_module()
class PointOBB(TwoStageDetector):
    def __init__(self,
                 backbone,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 construct_view = True,
                 construct_resize = False,
                 weight_mode = 'dynamic_weight',
                 scale_classes = None,
                 scale_classes2 = None,
                 loss_diff_view=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 crop_size = (1024, 1024),
                 padding = 'reflection',
                 view_range: Tuple[float, float] = (0.25, 0.75),
                 bbox_head=None,
                 bbox_pred_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointOBB, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_stages = roi_head.num_stages
        self.stage = 0
        
        print(f'========={self.stage}===========')
        if bbox_head is not None:
            self.with_bbox_head = True
            self.bbox_head = build_head(bbox_head)
        self.crop_size = crop_size
        self.padding = padding
        self.view_range = view_range
        self.loss_diff_view = build_loss(loss_diff_view)
        self.construct_view = construct_view
        self.construct_resize = construct_resize
        self.weight_mode  = weight_mode
        self.scale_classes = scale_classes
        self.scale_classes2 = scale_classes2

        if bbox_pred_head is not None:
            self.bbox_pred_head = build_head(bbox_pred_head)

        if train_cfg is not None:
            self.iter_count = train_cfg.get("iter_count")
            self.burn_in_steps1 = train_cfg.get("burn_in_steps1")
            self.burn_in_steps2 = train_cfg.get("burn_in_steps2")

    # my:  对图片做旋转和裁剪  输入：images  旋转角度  裁剪大小  实例  padding方式  输出：crop后的images和实例
    def rotate_crop(
            self,
            batch_inputs: Tensor,
            rot: float = 0.,
            size: Tuple[int, int] = (768, 768),
            batch_gt_instances = None,
            padding: str = 'reflection'):
        """

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            rot (float): Angle of view rotation. Defaults to 0.
            size (tuple[int]): Crop size from image center.
                Defaults to (768, 768).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  
                padding (str): Padding method of image black edge.
                Defaults to 'reflection'.

        Returns:
            Processed batch_inputs (Tensor) and batch_gt_instances
            (list[:obj:`InstanceData`])
        """
        device = batch_inputs.device
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2 
        crop_w = (w - size_w) // 2 
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = batch_inputs.new_tensor([[cosa, -sina], [sina, cosa]],
                                         dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device) 
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1]) 
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2) 
            # rotate
            batch_inputs = grid_sample(
                batch_inputs, grid, 'bilinear', padding, align_corners=True)
            if batch_gt_instances is not None:
                for i, gt_instances in enumerate(batch_gt_instances):
                    gt_bboxes = gt_instances
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[  
                        ..., 2:4], gt_bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + rot
                    rot_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                    batch_gt_instances[i] = rot_gt_bboxes
        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h,
                                    crop_w:crop_w + size_w]
        if batch_gt_instances is None:
            return batch_inputs
        else:  # rot == 0
            for i, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = gt_instances
                xy, wh, a = gt_bboxes[..., :2], gt_bboxes[...,
                                                          2:4], gt_bboxes[...,
                                                                          [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                batch_gt_instances[i] = crop_gt_bboxes

            return batch_inputs, batch_gt_instances

    # my:  构建rot/flip视图   输出：两个视图ori/rot的拼接
    def construct_Rview(self, img, generate_proposals_0, gt_bboxes, img_metas, 
                        gt_labels, gt_true_bboxes, gt_bboxes_ignore, proposals_valid_list_0):
        
        img_ori = img.clone()
        # 1) # Crop original images and gts
        batch_gt_bboxes = hboxlist2cxcywha(gt_bboxes)
        batch_proposals = hboxlist2cxcywha(generate_proposals_0)
        batch_instances_all, interval_flag = merge_batch_list(batch_gt_bboxes, batch_proposals)

        img, batch_instances_all = self.rotate_crop(img, 0, self.crop_size, batch_instances_all, self.padding)

        offset_gt = 1
        offset = 1
        for i, img_meta in enumerate(img_metas):
            img_meta['gt_bid'] = torch.arange(0, interval_flag[i][0], 1,
                device=batch_instances_all[i].device) + offset_gt + 0.2
            offset_gt += interval_flag[i][0]
            img_meta['bid'] = torch.arange(0, interval_flag[i][1], 1,
                device=batch_instances_all[i].device) + offset + 0.2
            offset += interval_flag[i][1]

        # 2) # Generate rotated images and gts
        rot = math.pi * ( 
            torch.rand(1, device=img.device) *
            (self.view_range[1] - self.view_range[0]) + self.view_range[0])
        batch_instance_rot = copy.deepcopy(batch_instances_all)
        img_metas_rot = copy.deepcopy(img_metas)
        img_rot, batch_instance_rot = self.rotate_crop(
            img, rot, self.crop_size, batch_instance_rot, self.padding)
        offset_gt = 1
        offset = 1
        for i, img_meta in enumerate(img_metas_rot):
            img_meta['gt_bid'] = torch.arange(0, interval_flag[i][0], 1,
                device=batch_instance_rot[i].device) + offset_gt + 0.4
            offset_gt += interval_flag[i][0]
            img_meta['bid'] = torch.arange(0, interval_flag[i][1], 1,
                device=batch_instance_rot[i].device) + offset + 0.4
            offset += interval_flag[i][1]

        # 3) # Generate flipped images and gts
        img_flp = transforms.functional.vflip(img)
        batch_instances_flp = copy.deepcopy(batch_instances_all)
        img_metas_flp = copy.deepcopy(img_metas)
        offset_gt = 1
        offset = 1
        for i, img_meta in enumerate(img_metas_flp):
            batch_instances_flp[i] = flip_tensor(batch_instances_flp[i], img.shape[2:4], 'vertical' )
            img_meta['gt_bid'] = torch.arange(0, interval_flag[i][0], 1,
                device=batch_instances_flp[i].device) + offset_gt + 0.6
            offset_gt += interval_flag[i][0]
            img_meta['bid'] = torch.arange(0, interval_flag[i][1], 1,
                device=batch_instances_flp[i].device) + offset + 0.6
            offset += interval_flag[i][1]

        # 4) # Concat original/rotated/flipped images and gts
        batch_gt_bboxes, batch_proposals = split_batch_list(batch_instances_all, interval_flag)
        batch_gt_bboxes_rot, batch_proposals_rot = split_batch_list(batch_instance_rot, interval_flag)
        batch_gt_bboxes_flp, batch_proposals_flp = split_batch_list(batch_instances_flp, interval_flag)

        proposals_valid_list_rot = []
        for v in range(len(proposals_valid_list_0)):
            rot_theta = batch_proposals_rot[v][:,-1].mean()
            w,h,_ = img_metas[v]['img_shape']
            img_xywha = batch_proposals_rot[v].new_tensor([w/2, h/2, w, h, rot_theta])  # (cx,cy,w,h,theta)
            iof_in_img = box_iou_rotated(batch_proposals_rot[v], img_xywha.unsqueeze(0), mode='iof')
            # iof_in_img = bbox_overlaps(pps_new.reshape(-1, 4), img_xyxy.unsqueeze(0), mode='iof')
            proposals_valid = iof_in_img > 0.8
            proposals_valid_list_rot.append(proposals_valid)

        img_ori, batch_instances_gt_true = self.rotate_crop(img_ori, 0, self.crop_size,gt_true_bboxes, self.padding)
        batch_instances_gt_true_rot = copy.deepcopy(batch_instances_gt_true)
        _, batch_instances_gt_true_rot = self.rotate_crop(img_ori, rot, self.crop_size, batch_instances_gt_true_rot, self.padding)
        batch_instances_gt_true_flp = copy.deepcopy(batch_instances_gt_true)
        for i, img_meta in enumerate(img_metas_flp):
            batch_instances_gt_true_flp[i] = flip_tensor(batch_instances_gt_true_flp[i], img_ori.shape[2:4], 'vertical' )
        
        batch_gt_bboxes_all = []
        batch_proposals_all = []
        img_metas_all = []
        gt_true_bboxes_all = []
        proposals_valid_list_all = []
        gt_labels_all = gt_labels + gt_labels
        gt_bboxes_ignore_all = gt_bboxes_ignore + gt_bboxes_ignore
        if torch.rand(1) < 0.95:
            img_inputs_all = torch.cat(
                (img, img_rot))
            for gt_box in batch_gt_bboxes + batch_gt_bboxes_rot:
                batch_gt_bboxes_all.append(gt_box)
            for proposal in batch_proposals + batch_proposals_rot:
                batch_proposals_all.append(proposal)
            for tmp_img_metas in img_metas + img_metas_rot:
                img_metas_all.append(tmp_img_metas)
            for gt_true in batch_instances_gt_true + batch_instances_gt_true_rot:
                gt_true_bboxes_all.append(gt_true)
            for proposal_valid in proposals_valid_list_0 + proposals_valid_list_rot:
                proposals_valid_list_all.append(proposal_valid)
        else:
            img_inputs_all = torch.cat(
                (img, img_flp))
            for gt_box in batch_gt_bboxes + batch_gt_bboxes_flp:
                batch_gt_bboxes_all.append(gt_box)
            for proposal in batch_proposals + batch_proposals_flp:
                batch_proposals_all.append(proposal)
            for tmp_img_metas in img_metas + img_metas_flp:
                img_metas_all.append(tmp_img_metas)
            for gt_true in batch_instances_gt_true + batch_instances_gt_true_flp:
                gt_true_bboxes_all.append(gt_true)
            for proposal_valid in proposals_valid_list_0 + proposals_valid_list_0:
                proposals_valid_list_all.append(proposal_valid)            

        return (img_inputs_all, batch_gt_bboxes_all, batch_proposals_all, img_metas_all,
                gt_labels_all, gt_true_bboxes_all, gt_bboxes_ignore_all, proposals_valid_list_all)
    
    # my： 计算两个视图间的得分相似性（SSC） 
    def Cross_View_Diff_Sim(self, results_v1, results_v2, gt_labels, proposals_valid, double_view, mode = 'scales', stage = 0):
        gt_label = torch.cat(gt_labels)
        base_proposal_cfg = self.train_cfg.get('base_proposal',self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',self.test_cfg.rpn)
        if mode == 'scales':
            num_base_scales = len(base_proposal_cfg['base_scales'])
        elif mode == 'ratios':
            num_base_scales = len(base_proposal_cfg['base_ratios'])
        elif mode == 'gts':
            num_base_scales = len(base_proposal_cfg['base_scales']) * len(base_proposal_cfg['base_ratios'])

        if stage >=1:
            if isinstance(fine_proposal_cfg['base_ratios'], tuple):
                num_base_scales = len(fine_proposal_cfg['base_ratios'][stage - 1])
            else:
                num_base_scales = len(fine_proposal_cfg['base_ratios'])
        if not double_view:
            v1_half_num = len(results_v1['cls_score'])
        else:
            v1_half_num = len(results_v1['cls_score'])//2
        cls_score_v1 = results_v1['cls_score'][:v1_half_num,...]
        ins_score_v1 = results_v1['ins_score'][:v1_half_num,...]
        # 取二者并集才是有效共同部分
        proposal_vaild = torch.cat(proposals_valid).reshape(cls_score_v1.size(0),-1,1)

        if stage < 1:
            cls_score_v1_prob = cls_score_v1.softmax(dim=-1)
        elif stage >= 1:
            cls_score_v1_prob = cls_score_v1.sigmoid()
        cls_score_v1_prob = cls_score_v1_prob * proposal_vaild
        ins_score_v1_prob = ins_score_v1.softmax(dim=1) * proposal_vaild
        # cls_score_v1_prob = cls_score_v1_prob
        # ins_score_v1_prob = ins_score_v1.softmax(dim=1)
        ins_score_v1_prob = F.normalize(ins_score_v1_prob, dim=1, p=1)
        prob_v1 = (cls_score_v1_prob * ins_score_v1_prob).sum(dim=1)

        cls_score_v2 = results_v2['cls_score']
        ins_score_v2 = results_v2['ins_score']
        
        if stage < 1:
            cls_score_v2_prob = cls_score_v2.softmax(dim=-1)
        elif stage >= 1:
            cls_score_v2_prob = cls_score_v2.sigmoid()
        cls_score_v2_prob = cls_score_v2_prob * proposal_vaild
        ins_score_v2_prob = ins_score_v2.softmax(dim=1)  * proposal_vaild
        ins_score_v2_prob = F.normalize(ins_score_v2_prob, dim=1, p=1)
        prob_v2 = (cls_score_v2_prob * ins_score_v2_prob).sum(dim=1)

        if stage>=1:
            cls_score_v1_prob_list = []
            cls_score_v2_prob_list = []
            ins_score_v1_prob_list = []
            ins_score_v2_prob_list = []
            for i in range(v1_half_num):
                cls_score_v1_prob_list.append(cls_score_v1_prob[i, ..., gt_label[i]].unsqueeze(0))
                cls_score_v2_prob_list.append(cls_score_v2_prob[i, ..., gt_label[i]].unsqueeze(0))
                ins_score_v1_prob_list.append(ins_score_v1_prob[i, ..., gt_label[i]].unsqueeze(0))
                ins_score_v2_prob_list.append(ins_score_v2_prob[i, ..., gt_label[i]].unsqueeze(0))
            cls_score_v1_prob = torch.cat(cls_score_v1_prob_list, dim=0)
            cls_score_v2_prob = torch.cat(cls_score_v2_prob_list, dim=0)
            ins_score_v1_prob = torch.cat(ins_score_v1_prob_list, dim=0)
            ins_score_v2_prob = torch.cat(ins_score_v2_prob_list, dim=0)
        
        cls_score_v1_prob = cls_score_v1_prob.reshape(cls_score_v1.size(0), num_base_scales, -1)
        ins_score_v1_prob = ins_score_v1_prob.reshape(ins_score_v1.size(0), num_base_scales, -1)
        cls_score_v2_prob = cls_score_v2_prob.reshape(cls_score_v2.size(0), num_base_scales, -1)
        ins_score_v2_prob = ins_score_v2_prob.reshape(ins_score_v2.size(0), num_base_scales, -1)
 
        cls_similarity = 1 - F.cosine_similarity(cls_score_v1_prob, cls_score_v2_prob, dim=-1, eps=1e-6)
        ins_similarity = 1 - F.cosine_similarity(ins_score_v1_prob, ins_score_v2_prob, dim=-1, eps=1e-6)
        score_similarity = 1 - F.cosine_similarity(prob_v1, prob_v2, dim=1, eps=1e-6)
        
        return cls_similarity, ins_similarity, score_similarity

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_true_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        self.e2e = True
        # stage1进入策略
        if self.iter_count == self.burn_in_steps1:
            self.roi_head.use_angle_loss = True
            print(f'#####iter_count1 use_angle_loss:{self.iter_count}#####')
            if self.construct_resize:
                self.construct_resize = False
        # 用于处理断点重训时，roi_head.use_angle_loss未被继承的问题
        if self.iter_count > self.burn_in_steps1:
            self.roi_head.use_angle_loss = True
            if self.construct_resize:
                self.construct_resize = False
        # stage2进入策略
        if self.iter_count == self.burn_in_steps2:
            if self.roi_head.use_angle_loss:
                self.roi_head.add_angle_pred_begin = True
                print(f'#####iter_count2 add_angle_pred_begin:{self.iter_count}#####')
        # 用于处理断点重训时，roi_head.add_angle_pred_begin未被继承的问题
        if self.iter_count > self.burn_in_steps2:
            if self.roi_head.use_angle_loss:
                self.roi_head.add_angle_pred_begin = True

        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()
        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]

        # my: CBP
        if self.stage == 0:
            # my: 生成初始框
            generate_proposals_0, proposals_valid_list_0 = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta = img_metas)
            
            # Construst Rotate/Filp View
            (img_inputs_all, batch_gt_bboxes_all, batch_proposals_all, img_metas_all,
            gt_labels_all, gt_true_bboxes_all, gt_bboxes_ignore_all, 
            proposals_valid_all) = self.construct_Rview(
                        img, generate_proposals_0, gt_bboxes, img_metas, gt_labels, 
                        gt_true_bboxes, gt_bboxes_ignore, proposals_valid_list_0)
            half_num = len(img_inputs_all)//2
            double_view = True
            if self.construct_view: 
                img = img_inputs_all
                gt_bboxes = batch_gt_bboxes_all
                generate_proposals_0 = batch_proposals_all
                proposals_valid_list_0 = proposals_valid_all
                img_metas = img_metas_all
                gt_labels = gt_labels_all
                gt_true_bboxes = gt_true_bboxes_all
                gt_bboxes_ignore = gt_bboxes_ignore_all

                if self.roi_head.not_use_rot_mil:
                    for p in range(half_num, len(proposals_valid_list_0)):
                        proposals_valid_list_0[p] = torch.zeros_like(proposals_valid_list_0[p],dtype=proposals_valid_list_0[0].dtype)
                
                if not self.roi_head.use_angle_loss: 
                    img = img_inputs_all[:half_num]
                    gt_bboxes = batch_gt_bboxes_all[:half_num]
                    generate_proposals_0 = batch_proposals_all[:half_num]
                    proposals_valid_list_0 = proposals_valid_all[:half_num]
                    img_metas = img_metas_all[:half_num]
                    gt_labels = gt_labels_all[:half_num]
                    gt_true_bboxes = gt_true_bboxes_all[:half_num]
                    gt_bboxes_ignore = gt_bboxes_ignore_all[:half_num]
                    double_view = False
            else: 
                img = img_inputs_all[:half_num]
                gt_bboxes = batch_gt_bboxes_all[:half_num]
                generate_proposals_0 = batch_proposals_all[:half_num]
                proposals_valid_list_0 = proposals_valid_all[:half_num]
                img_metas = img_metas_all[:half_num]
                gt_labels = gt_labels_all[:half_num]
                gt_true_bboxes = gt_true_bboxes_all[:half_num]
                gt_bboxes_ignore = gt_bboxes_ignore_all[:half_num]
                double_view = False
            
            gt_points_all = [b[:, :2] for b in gt_bboxes]
            feat = self.extract_feat(img)
            generate_proposals_init = copy.deepcopy(generate_proposals_0)
            max_roi_num = 10000 ## control roi num, avoid OOM
            # max_roi_num = 7000
            
            for i in range(len(gt_points_all)):
                gt_num = gt_points_all[i].size(0)
                proposals_num = len(proposals_valid_list_0[i])
                
                if proposals_num > max_roi_num: 
                    num1 = proposals_valid_list_0[i].size(-1)
                    num2 = generate_proposals_0[i].size(-1)
                    vaild_range = torch.arange(gt_num, device=img.device)
                    num_roi_per_gt = int(proposals_num / gt_num)
                    max_gt_num = max_roi_num // num_roi_per_gt
                    select_inds = torch.randperm(vaild_range.numel())[:max_gt_num].to(device=img.device)
                    proposals_valid_list_0[i] = proposals_valid_list_0[i].reshape(gt_num, -1, num1)[select_inds].reshape(-1, num1)
                    generate_proposals_0[i] = generate_proposals_0[i].reshape(gt_num, -1, num2)[select_inds].reshape(-1, num2)
                    gt_true_bboxes[i] = gt_true_bboxes[i][select_inds]
                    gt_points_all[i] = gt_points_all[i][select_inds]
                    gt_labels[i] = gt_labels[i][select_inds]
                    gt_bboxes[i] = gt_bboxes[i][select_inds]

            dynamic_weight_init = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
            # my：送进roi_head里计算损失   返回：MIL_loss， ins&cls得分， 最后筛选融合的伪框， 动态权重
            roi_losses_0, bbox_results, pseudo_boxes, dynamic_weight = self.roi_head.forward_train(self.stage, feat, img_metas_all,
                                                                                                    gt_points_all,
                                                                                                    generate_proposals_0,
                                                                                                    generate_proposals_0,
                                                                                                    proposals_valid_list_0,
                                                                                                    None, None,
                                                                                                    gt_true_bboxes, gt_labels,
                                                                                                    dynamic_weight_init, False,
                                                                                                    gt_bboxes_ignore, gt_masks,
                                                                                                    **kwargs)

            ## 添加随机缩放
            if self.construct_resize:
                resize = 0.5 + 1.0 * torch.rand(1, dtype=torch.float32, device=img.device).item()
                img_dview = resize_image(img[:half_num], resize_ratio=resize)
                generate_proposals_dview = generate_proposals_0[:half_num] 
                gt_bboxes_dview = gt_bboxes[:half_num]
                gt_true_bboxes_dview = gt_true_bboxes[:half_num]
                proposals_valid_list_dview = proposals_valid_list_0[:half_num]
                gt_labels_dview = gt_labels[:half_num]

                img_metas_dview, gt_bboxes_dview, generate_proposals_dview, gt_true_bboxes_dview,\
                    gt_bboxes_ignore_dview = resize_rotate_proposal(img_metas_all[:half_num], 
                                                                    gt_bboxes_dview,
                                                                    generate_proposals_dview, 
                                                                    gt_true_bboxes_dview, 
                                                                    gt_bboxes_ignore_all[:half_num], ratio = resize)
                gt_points_dview = [b[:, :2] for b in gt_bboxes_dview]
                feat_dview = self.extract_feat(img_dview)
                dynamic_weight_dview = torch.cat(gt_labels_dview).new_ones(len(torch.cat(gt_labels_dview)))
                roi_losses_dview, bbox_results_dview, pseudo_boxes_dview, \
                    dynamic_weight_dview = self.roi_head.forward_train(self.stage, feat_dview, img_metas_dview,
                                                                       gt_points_dview,
                                                                       generate_proposals_dview,
                                                                       generate_proposals_dview,
                                                                       proposals_valid_list_dview,
                                                                       None, None,
                                                                       gt_true_bboxes_dview, gt_labels_dview,
                                                                       dynamic_weight_dview, True,
                                                                       gt_bboxes_ignore_dview, gt_masks,
                                                                       **kwargs)
                cls_sim, ins_sim, score_sim = self.Cross_View_Diff_Sim(bbox_results, bbox_results_dview, gt_labels_all[:half_num],
                                                                       proposals_valid_list_dview, double_view,
                                                                       mode = 'scales', stage=0)
                                                                    #  mode = 'ratios', stage=0)
                                                                    #  mode = 'gts', stage=0)
                                                                                        
                                                                
                loss_scale1 = 1.0 * self.loss_diff_view(cls_sim, torch.zeros_like(cls_sim))  
                loss_scale2 = 2.0 * self.loss_diff_view(ins_sim, torch.zeros_like(ins_sim))
                losses[f'stage{self.stage}_loss_SSC_cls'] = loss_scale1
                losses[f'stage{self.stage}_loss_SSC_ins'] = loss_scale2


                for key, value in roi_losses_dview.items():
                    losses[f'stage{self.stage}_dview_{key}'] = value
            
            for key, value in roi_losses_0.items():
                losses[f'stage{self.stage}_{key}'] = value
                
            self.stage +=1
            del generate_proposals_0
            del proposals_valid_list_0
            del roi_losses_0

        stage_remain = self.num_stages - self.stage
        for re in range(stage_remain):
            generate_proposals, proposals_valid_list = fine_rotate_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                img_meta=img_metas,
                                                                                stage=self.stage)
            generate_proposals_all = generate_proposals_init + generate_proposals
            neg_proposal_list, neg_weight_list = gen_rotate_negative_proposals(gt_points_all, fine_proposal_cfg,
                                                                               generate_proposals_all,
                                                                               img_meta=img_metas)
            if self.roi_head.use_angle_loss and self.roi_head.not_use_rot_mil:
                half_num = len(proposals_valid_list)//2
                for p in range(half_num, len(proposals_valid_list)):
                    proposals_valid_list[p] = torch.zeros_like(proposals_valid_list[p],
                                                               dtype=proposals_valid_list[0].dtype)

            roi_losses_i, bbox_results_i, pseudo_boxes, dynamic_weight = self.roi_head.forward_train(self.stage, feat, img_metas,
                                                                                    gt_points_all,
                                                                                    pseudo_boxes,
                                                                                    generate_proposals,
                                                                                    proposals_valid_list,
                                                                                    neg_proposal_list, neg_weight_list,
                                                                                    gt_true_bboxes, gt_labels,
                                                                                    dynamic_weight, False,
                                                                                    gt_bboxes_ignore, gt_masks,
                                                                                    **kwargs)
            
            for key, value in roi_losses_i.items():
                losses[f'stage{self.stage}_{key}'] = value
            self.stage +=1
            del roi_losses_i
        
        if self.e2e:
            # ins_score weight
            ins_score_weight = []
            for x in range(half_num):
                ins_score = bbox_results_i["filtered_scores"][x]["ins_score"].sum(-1).detach()
                ins_score_weight.append(ins_score)
            ins_score_weight = torch.cat(ins_score_weight)
                        
            if self.weight_mode == 'ins_score':
                weight_to_use = ins_score_weight
            if self.weight_mode == 'dynamic_weight':
                weight_to_use = dynamic_weight
            split_weights = []
            start_idx = 0
            for boxes in pseudo_boxes[:half_num]:
                num_gts = boxes.shape[0]  
                end_idx = start_idx + num_gts 
                split_weights.append(weight_to_use[start_idx:end_idx])
                start_idx = end_idx
                
            
            feat_half = [f[:half_num] for f in feat]
            cls_scores, bbox_preds, angle_preds, centernesses = self.bbox_pred_head.forward(feat_half)
        
            loss_e2e = self.bbox_pred_head.loss(cls_scores, bbox_preds, angle_preds, 
                                                centernesses, pseudo_boxes[:half_num], 
                                                gt_labels[:half_num], split_weights, img_metas[:half_num])
            losses['loss_e2e_cls'] = loss_e2e['loss_cls']
            losses['loss_e2e_bbox'] = loss_e2e['loss_bbox']
            losses['loss_e2e_centerness'] = loss_e2e['loss_centerness'] 
          
        
        if self.stage > 1:
            for j in range(len(gt_points)):
                del generate_proposals[0]
                del proposals_valid_list[0]
                del pseudo_boxes[0]
            del dynamic_weight
            generate_proposals.clear()
            proposals_valid_list.clear()
            pseudo_boxes.clear()

        torch.cuda.empty_cache()
        self.stage = 0

        self.iter_count += 1
        return losses
    
    # e2e test
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_pred_head.forward(x)
        bbox_list = self.bbox_pred_head.get_bboxes(
            *outs, img_metas, rescale=rescale, cfg=self.test_cfg)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_pred_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        
        if self.scale_classes:
            for id in self.scale_classes:
                bbox_results = scale_bbox(bbox_results, id)
        if self.scale_classes2:
            for id in self.scale_classes2:
                bbox_results = scale_bbox2(bbox_results, id)

        return bbox_results

        
       
    




