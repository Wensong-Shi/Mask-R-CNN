from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align

import cfg
import utils.bbox as bbox
from models import backbone, rpn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProposalTargetCreator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, proposals, gt_labels, gt_bboxes, gt_masks):
        batch_size = len(proposals)
        proposals_list = []
        labels_list = []
        reg_list = []
        masks_target_list = []
        for i in range(batch_size):
            overlap = box_iou(proposals[i], gt_bboxes[i])
            overlap, index_gt = torch.max(overlap, dim=1)
            index_fg = overlap >= cfg.HI_OVERLAP_ROI_TRAIN
            """index_lo = overlap <= cfg.LO_OVERLAP_ROI_TRAIN
            index_bg = ~(index_fg + index_lo)"""
            index_bg = overlap < cfg.HI_OVERLAP_ROI_TRAIN
            assert sum(index_fg) + sum(index_bg) >= cfg.NUM_SAMPLES_ROI_TRAIN, 'Number of RoI samples if wrong!'
            if sum(index_bg) < cfg.NUM_SAMPLES_ROI_TRAIN * (1 - cfg.FG_FRACTION_ROI_TRAIN):
                num_fg = int(cfg.NUM_SAMPLES_ROI_TRAIN - sum(index_bg))
            else:
                num_fg = cfg.NUM_SAMPLES_ROI_TRAIN * cfg.FG_FRACTION_ROI_TRAIN
            # 选择正样本
            if sum(index_fg) > num_fg:
                index_proposals = torch.arange(len(proposals[i]), device=device)
                index_proposals = index_proposals[index_fg]
                keep = list(range(len(index_proposals)))
                shuffle(keep)
                keep = keep[:num_fg]
                keep = index_proposals[keep]
                proposals_positive = proposals[i][keep]
                labels_positive = gt_labels[i][index_gt][keep]
                reg = bbox.bbox2reg(proposals_positive, gt_bboxes[i][index_gt][keep])
                intersection = bbox.intersection(proposals_positive, gt_bboxes[i][index_gt][keep])
                masks_target = self.get_masks_target(gt_masks[i][index_gt][keep], intersection)
            else:
                proposals_positive = proposals[i][index_fg]
                labels_positive = gt_labels[i][index_gt][index_fg]
                reg = bbox.bbox2reg(proposals_positive, gt_bboxes[i][index_gt][index_fg])
                intersection = bbox.intersection(proposals_positive, gt_bboxes[i][index_gt][index_fg])
                masks_target = self.get_masks_target(gt_masks[i][index_gt][index_fg], intersection)
            # 选择负样本
            num_bg = cfg.NUM_SAMPLES_ROI_TRAIN - len(proposals_positive)
            if sum(index_bg) > num_bg:
                index_proposals = torch.arange(len(proposals[i]), device=device)
                index_proposals = index_proposals[index_bg]
                keep = list(range(len(index_proposals)))
                shuffle(keep)
                keep = keep[:num_bg]
                keep = index_proposals[keep]
                proposals_negative = proposals[i][keep]
            else:
                proposals_negative = proposals[i][index_bg]
            labels_negative = torch.zeros([len(proposals_negative)], dtype=torch.uint8, device=device)
            # cat
            proposals_sample = torch.cat([proposals_positive, proposals_negative], dim=0)
            labels_sample = torch.cat([labels_positive, labels_negative], dim=0)
            proposals_list.append(proposals_sample)
            labels_list.append(labels_sample)
            reg_list.append(reg)
            masks_target_list.append(masks_target)
        proposals_batch = torch.stack(proposals_list)
        labels_batch = torch.stack(labels_list)
        """
        proposals_batch: B * NUM_SAMPLES_ROI * 4, [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...], ...], torch.float32
        labels_batch: B * NUM_SAMPLES_ROI, [[label1, label2, ..., 0, 0, ...], ...], torch.uint8
        reg_list: B * (num_positive * 4), [[[dx, dy, dw, dh], [dx, dy, dw, dh], ...], ...], list(torch.float32)
        masks_target_list: B * (num_positive * 28 * 28), list(torch.int32)
        """
        return proposals_batch, labels_batch, reg_list, masks_target_list

    @staticmethod
    def get_masks_target(gt_masks_i, intersection):
        # format
        gt_masks_i = torch.stack([gt_masks_i], dim=1)
        intersection = torch.cat([torch.arange(gt_masks_i.size()[0], device=device).view(-1, 1), intersection], dim=1)
        # get masks target
        masks_target = roi_align(gt_masks_i.float(), intersection, 28, 1)
        return masks_target.view(-1, 28, 28).round().int()  # num_positive * 28 * 28, torch.int32

    def backward(self):
        pass


class MaskRCNN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ['TRAIN', 'EVAL'], 'Wrong mode!'
        self.mode = mode
        self.backbone = backbone.Resnet101FPN(self.mode)
        self.rpn = rpn.RPN(self.mode)
        # head
        self.head_faster_rcnn = nn.Sequential(nn.Linear(256 * 7 * 7, 1024), nn.ReLU(True),
                                              nn.Linear(1024, 1024), nn.ReLU(True))
        self.head_mask_rcnn = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
                                            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
                                            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
                                            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
                                            nn.ConvTranspose2d(256, 256, 2, 2, 0), nn.ReLU(True))
        # output
        self.fc_cls = nn.Linear(1024, cfg.NUM_CLASSES)
        self.fc_reg = nn.Linear(1024, cfg.NUM_CLASSES * 4)
        self.conv_mask = nn.Conv2d(256, cfg.NUM_CLASSES, 1, 1, 0)
        # initialize
        if self.mode == 'TRAIN':
            self.initialize()
            self.proposal_target_creator = ProposalTargetCreator()

    def forward(self, img, img_info, gt_labels, gt_bboxes, gt_masks):
        batch_size = img.size()[0]
        features = self.backbone(img)
        proposals, loss_rpn_cls, loss_rpn_reg = self.rpn(features, img_info, gt_bboxes)
        if self.mode == 'TRAIN':
            proposals, labels_target, reg_target, masks_target = self.proposal_target_creator(proposals, gt_labels, gt_bboxes, gt_masks)
        # format proposals
        batch_index = torch.zeros([batch_size, proposals.size()[1], 1], device=device)
        for i in range(batch_size):
            batch_index[i].fill_(i)
        proposals = torch.cat([batch_index, proposals], dim=2).view(-1, 5)
        # map proposals
        h_proposals = proposals[:, 4] - proposals[:, 2] + 1
        w_proposals = proposals[:, 3] - proposals[:, 1] + 1
        level_proposals = torch.log2(torch.sqrt(h_proposals * w_proposals) / 56)
        level_proposals = torch.floor(level_proposals)
        level_proposals = level_proposals.clamp(0, 3).int()
        # forward(train)
        if self.mode == 'TRAIN':
            # roi align
            features_pooled_fc = features[0].new_zeros([proposals.size()[0], 256, 7, 7])
            features_pooled_conv = features[0].new_zeros([proposals.size()[0], 256, 14, 14])
            for i, stride in zip(range(4), cfg.FEATURE_STRIDES[:4]):
                index = level_proposals == i
                if index.any():
                    features_pooled_fc[index] = roi_align(features[i], proposals[index], 7, 1 / stride)
                    features_pooled_conv[index] = roi_align(features[i], proposals[index], 14, 1 / stride)
            # head
            features_pooled_fc = self.head_faster_rcnn(features_pooled_fc.view(-1, 256 * 7 * 7))
            features_pooled_conv = self.head_mask_rcnn(features_pooled_conv)
            # output
            prob = self.fc_cls(features_pooled_fc)
            prob = F.softmax(prob, dim=1)
            x_reg = self.fc_reg(features_pooled_fc)
            masks_pred = self.conv_mask(features_pooled_conv)
            # output loss
            loss_output_cls = F.cross_entropy(prob, labels_target.view(-1))
            # format reg
            x_reg = x_reg.view(-1, cfg.NUM_CLASSES, 4)
            x_reg = torch.gather(x_reg, 1, labels_target.view(-1, 1, 1).expand(-1, -1, 4).long())
            x_reg = x_reg.view(batch_size, -1, 4)
            # format mask
            masks_pred = masks_pred.view(-1, cfg.NUM_CLASSES, 28 * 28)
            masks_pred = torch.gather(masks_pred, 1, labels_target.view(-1, 1, 1).expand(-1, -1, 28 * 28).long())
            masks_pred = masks_pred.view(batch_size, -1, 28, 28)
            loss_output_reg_list = []
            loss_output_mask_list = []
            for i in range(batch_size):
                if sum(labels_target[i] > 0) == 0:
                    loss_output_reg_list.append(0)
                    loss_output_mask_list.append(0)
                else:
                    loss_output_reg_list.append(F.smooth_l1_loss(x_reg[i][labels_target[i] > 0], reg_target[i]))
                    loss_output_mask_list.append(F.binary_cross_entropy_with_logits(masks_pred[i][labels_target[i] > 0],
                                                                                    masks_target[i].float()))
            loss_output_reg = sum(loss_output_reg_list) / batch_size
            loss_output_mask = sum(loss_output_mask_list) / batch_size
            loss = loss_rpn_cls + loss_rpn_reg + loss_output_cls + loss_output_reg + loss_output_mask
            return loss  # tensor(loss, dtype=torch.float32)
        # forward(eval)


    def initialize(self):
        for layer in [self.head_faster_rcnn[0], self.head_faster_rcnn[2], self.head_mask_rcnn[0],
                      self.head_mask_rcnn[2], self.head_mask_rcnn[4], self.head_mask_rcnn[6], self.head_mask_rcnn[8],
                      self.fc_cls, self.fc_reg, self.conv_mask]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias != None:
                nn.init.constant_(layer.bias, 0)

    def set_train(self):
        self.train()
        # 因为batch size较小，将bn层设置为eval模式
        for layer in [self.backbone.base_layer0, self.backbone.base_layer1, self.backbone.base_layer2,
                      self.backbone.base_layer3, self.backbone.base_layer4]:
            layer.apply(self.set_bn_eval)

    @staticmethod
    def set_bn_eval(layer):
        name = layer.__class__.__name__
        if name.find('BatchNorm') != -1:
            layer.eval()
