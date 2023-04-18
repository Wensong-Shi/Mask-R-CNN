from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_iou

import cfg
import utils.bbox as bbox
from utils.anchor import generate_anchor


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProposalCreator(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ['TRAIN', 'EVAL'], 'Wrong mode!'
        self.mode = mode
        if self.mode == 'TRAIN':
            self.top_n_pre_nms = cfg.TOP_N_PRE_NMS_RPN_TRAIN
            self.top_n_post_nms = cfg.TOP_N_POST_NMS_RPN_TRAIN
            self.thresh_nms = cfg.THRESH_NMS_RPN_TRAIN
        else:
            self.top_n_pre_nms = cfg.TOP_N_PRE_NMS_RPN_TEST
            self.top_n_post_nms = cfg.TOP_N_POST_NMS_RPN_TEST
            self.thresh_nms = cfg.THRESH_NMS_RPN_TEST

    def forward(self, prob_list, reg_list, feature_shapes, img_info):
        batch_size = prob_list[0].size()[0]
        proposals_batch = prob_list[0].new_zeros([batch_size, self.top_n_post_nms, 4])
        for i in range(batch_size):
            score_list = []
            proposals_list = []
            for prob, reg, feature_shape, feature_stride, anchor_size in zip(prob_list, reg_list, feature_shapes, cfg.FEATURE_STRIDES, cfg.ANCHOR_SIZES):
                anchors = generate_anchor(feature_shape, feature_stride, anchor_size, cfg.ANCHOR_RATIOS)
                proposals = bbox.reg2bbox(anchors, reg[i])
                prob_i, proposals = bbox.clip_bbox(prob[i], proposals, img_info[i])
                # pre nms
                if len(prob_i) > self.top_n_pre_nms > 0:
                    _, index = torch.sort(prob_i, dim=0, descending=True)
                    index = index[:self.top_n_pre_nms]
                    prob_i = prob_i[index]
                    proposals = proposals[index]
                # nms
                index = nms(proposals, prob_i, self.thresh_nms)
                # post nms
                if len(index) > self.top_n_post_nms > 0:
                    index = index[:self.top_n_post_nms]
                prob_i = prob_i[index]
                proposals = proposals[index]
                score_list.append(prob_i)
                proposals_list.append(proposals)
            # multi levels merge
            score = torch.cat(score_list, dim=0)
            proposals = torch.cat(proposals_list, dim=0)
            if len(score) > self.top_n_post_nms > 0:
                _, index = torch.sort(score, dim=0, descending=True)
                index = index[:self.top_n_post_nms]
                proposals = proposals[index]
            proposals_batch[i, :len(proposals), :] = proposals
            # 防止box_iou函数出错
            if len(proposals) < self.top_n_post_nms:
                proposals_batch[i, len(proposals):, 2] = 1.
                proposals_batch[i, len(proposals):, 3] = 1.
        # B * TOP_N_POST_NMS * 4, [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...], ...], torch.float32
        return proposals_batch

    def backward(self):
        pass


class AnchorTargetCreator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_shapes, gt_bboxes):
        batch_size = len(gt_bboxes)
        anchors_list = []
        for feature_shape, feature_stride, anchor_size in zip(feature_shapes, cfg.FEATURE_STRIDES, cfg.ANCHOR_SIZES):
            anchors = generate_anchor(feature_shape, feature_stride, anchor_size, cfg.ANCHOR_RATIOS)
            anchors_list.append(anchors)
        anchors = torch.cat(anchors_list, dim=0)
        # 为每个anchor打上标签，-1: ignore, 0: negative, 1: positive
        labels = torch.zeros([batch_size, len(anchors)], dtype=torch.int8).fill_(-1)
        reg_list = []
        for i in range(batch_size):
            overlap = box_iou(anchors, gt_bboxes[i])
            # 设置与各个gt_bbox的iou最大的anchor为正样本
            _, index = torch.max(overlap, dim=0)
            labels[i][index] = 1
            num_fg = int(cfg.NUM_SAMPLES_RPN_TRAIN * cfg.FG_FRACTION_RPN_TRAIN)
            if sum(labels[i] == 1) > num_fg:
                drop = list(range(len(index)))
                shuffle(drop)
                drop = drop[num_fg:]
                index = index[drop]
                labels[i][index] = -1
            # 选择正样本
            overlap, index_gt = torch.max(overlap, dim=1)
            index = overlap > cfg.OVERLAP_POSITIVE_RPN_TRAIN
            if (num_fg - sum(labels[i] == 1)) > 0:
                if sum(index) > (num_fg - sum(labels[i] == 1)):
                    index_anchor = torch.arange(len(anchors), device=device)
                    index_anchor = index_anchor[index]
                    keep = list(range(len(index_anchor)))
                    shuffle(keep)
                    keep = keep[:int(num_fg - sum(labels[i] == 1))]
                    keep = index_anchor[keep]
                    labels[i][keep] = 1
                else:
                    labels[i][index] = 1
            # 选择负样本
            num_bg = int(cfg.NUM_SAMPLES_RPN_TRAIN - sum(labels[i] == 1))
            index_hi = overlap >= cfg.HI_OVERLAP_NEGATIVE_RPN_TRAIN
            index_lo = overlap <= cfg.LO_OVERLAP_NEGATIVE_RPN_TRAIN
            index = ~(index_hi + index_lo)
            assert sum(index) >= num_bg, 'Number of RPN samples is wrong!'
            if sum(index) > num_bg:
                index_anchor = torch.arange(len(anchors), device=device)
                index_anchor = index_anchor[index]
                keep = list(range(len(index_anchor)))
                shuffle(keep)
                keep = keep[:num_bg]
                keep = index_anchor[keep]
                labels[i][keep] = 0
            else:
                labels[i][index] = 0
            reg = bbox.bbox2reg(anchors[labels[i] == 1], gt_bboxes[i][index_gt][labels[i] == 1])
            reg_list.append(reg)
        """
        labels: B * num_anchors, [[-1, -1, 1, 0, -1, ...], ...], torch.int8
        reg_list: B * (num_positive * 4), [[[dx, dy, dw, dh], [dx, dy, dw, dh], ...], ...], list(torch.float32)
        """
        return labels.cuda() if torch.cuda.is_available() else labels, reg_list

    def backward(self):
        pass


class RPN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ['TRAIN', 'EVAL'], 'Wrong mode!'
        self.mode = mode
        self.rpn_conv = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True))
        self.out_channels_cls = len(cfg.ANCHOR_RATIOS) * 1
        self.out_channels_reg = len(cfg.ANCHOR_RATIOS) * 4
        self.rpn_conv_cls = nn.Conv2d(512, self.out_channels_cls, 1, 1, 0)
        self.rpn_conv_reg = nn.Conv2d(512, self.out_channels_reg, 1, 1, 0)
        self.proposal_creator = ProposalCreator(self.mode)
        # initialize
        if self.mode == 'TRAIN':
            self.initialize()
            self.anchor_target_creator = AnchorTargetCreator()

    def forward(self, features, img_info, gt_bboxes):
        batch_size = features[0].size()[0]
        feature_shapes = []
        cls_list = []
        prob_list = []
        reg_list = []
        for x in features:
            feature_shapes.append([x.size()[2], x.size()[3]])
            x = self.rpn_conv(x)
            x_cls = self.rpn_conv_cls(x)
            x_reg = self.rpn_conv_reg(x)
            # format x
            x_cls = x_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
            x_reg = x_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            prob = x_cls.sigmoid()
            cls_list.append(x_cls)
            prob_list.append(prob)
            reg_list.append(x_reg)
        # get proposals
        proposals = self.proposal_creator(prob_list, reg_list, feature_shapes, img_info)
        # RPN loss
        loss_rpn_cls = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        loss_rpn_reg = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        if self.mode == 'TRAIN':
            labels_anchor, reg_target = self.anchor_target_creator(feature_shapes, gt_bboxes)
            x_cls = torch.cat(cls_list, dim=1)
            x_reg = torch.cat(reg_list, dim=1)
            loss_rpn_cls_list = []
            loss_rpn_reg_list = []
            for i in range(batch_size):
                loss_rpn_cls_list.append(F.binary_cross_entropy_with_logits(x_cls[i][labels_anchor[i] >= 0],
                                                                            labels_anchor[i][labels_anchor[i] >= 0].float()))
                loss_rpn_reg_list.append(F.smooth_l1_loss(x_reg[i][labels_anchor[i] == 1], reg_target[i]))
            loss_rpn_cls = sum(loss_rpn_cls_list) / batch_size
            loss_rpn_reg = sum(loss_rpn_reg_list) / batch_size
        """
        proposals: B * TOP_N_POST_NMS * 4, [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...], ...], torch.float32
        loss_rpn_cls, loss_rpn_reg: tensor(loss, dtype=torch.float32)
        """
        return proposals, loss_rpn_cls, loss_rpn_reg

    def initialize(self):
        for layer in [self.rpn_conv[0], self.rpn_conv_cls, self.rpn_conv_reg]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias != None:
                nn.init.constant_(layer.bias, 0)
