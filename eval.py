import json
import logging

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

import cfg
from coco_dataloader import dataset_coco, collate_fn_coco
from models.mask_rcnn import MaskRCNN


device = 'cuda' if torch.cuda.is_available() else 'cpu'


mode = ['TRAIN', 'EVAL'][1]
dataset_type = ['train2017', 'val2017', 'test2017'][1]
path_checkpoints = 'train_backup/test.pth'

# prepare base things
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(cfg.LOGFILE_TEST), logging.StreamHandler()])
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# prepare dataloader
dataset = dataset_coco.DatasetCoco(mode, dataset_type, cfg.DATASET_ROOT_DIR, cfg.SHORT_SIDE, cfg.IMAGE_NORMALIZE_INFO)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS,
                        collate_fn=collate_fn_coco.collate_fn_coco, pin_memory=cfg.PIN_MEMORY)
# prepare model
model = MaskRCNN(mode)
if use_cuda:
    model.cuda()
checkpoints = torch.load(path_checkpoints)
model.load_state_dict(checkpoints)
logging.info(f'Load weights from {path_checkpoints}.')
# eval
logging.info('Start evaluating!')
model.eval()
with torch.no_grad():
    img_ids = []
    results = []
    for batch_index, data in enumerate(dataloader):
        logging.info(f'Predict {batch_index + 1}/{len(dataloader)}.')
        img_id, img, img_info, gt_labels, gt_bboxes, gt_masks = data
        if use_cuda:
            for i in range(len(gt_bboxes)):
                gt_labels[i] = gt_labels[i].cuda()
                gt_bboxes[i] = gt_bboxes[i].cuda()
                gt_masks[i] = gt_masks[i].cuda()
        prob, labels_pred, bboxes_pred, masks_pred = model(img.type(FloatTensor), img_info, gt_labels, gt_bboxes,
                                                           gt_masks)
        # process results
        index_bboxes = torch.arange(len(prob), device=device)
        _, index_mask2bbox = torch.sort(prob, descending=True)
        index_mask2bbox = index_mask2bbox[:100]
        # 逐个类别处理结果
        for i in range(1, cfg.NUM_CLASSES):
            index = labels_pred == i
            if index.any():
                prob_i = prob[index]
                labels_pred_i = labels_pred[index]
                bboxes_pred_i = bboxes_pred[index]
                index_bboxes_i = index_bboxes[index]
                index = prob_i > 0.5  # 过滤掉得分较低的bboxes
                if index.any():
                    prob_i = prob_i[index]
                    labels_pred_i = labels_pred_i[index]
                    bboxes_pred_i = bboxes_pred_i[index]
                    index_bboxes_i = index_bboxes_i[index]
                    # nms
                    index = nms(bboxes_pred_i, prob_i, cfg.THRESH_NMS_RPN_TEST)
                    prob_i = prob_i[index]
                    labels_pred_i = labels_pred_i[index]
                    bboxes_pred_i = bboxes_pred_i[index]
                    index_bboxes_i = index_bboxes_i[index]
                    for prob_i_j, label_pred_i_j, bbox_pred_i_j, index_bbox_i_j in zip(prob_i, labels_pred_i,
                                                                                       bboxes_pred_i, index_bboxes_i):
                        if index_bbox_i_j in index_mask2bbox:
                            h, w, scale_factor = img_info[0]
                            h = (h * scale_factor).round()
                            w = (w * scale_factor).round()
                            mask_pred_j = masks_pred[index_mask2bbox == index_bbox_i_j][0]
                            # get label result
                            label_pred_i_j = dataset.dict_clsid2catid[int(label_pred_i_j)]
                            # get bbox result
                            bbox_pred_i_j *= scale_factor
                            x1, y1, x2, y2 = bbox_pred_i_j
                            bbox_pred_i_j[2] = bbox_pred_i_j[2] - bbox_pred_i_j[0] + 1
                            bbox_pred_i_j[3] = bbox_pred_i_j[3] - bbox_pred_i_j[1] + 1
                            # get mask result
                            w_bbox = bbox_pred_i_j[2].ceil()
                            h_bbox = bbox_pred_i_j[3].ceil()
                            mask_pred_j = np.array(mask_pred_j.cpu())
                            mask_pred_j = cv.resize(mask_pred_j, [int(w_bbox), int(h_bbox)],
                                                    interpolation=cv.INTER_LINEAR)
                            _, mask_pred_j = cv.threshold(mask_pred_j, 0.5, 1, cv.THRESH_BINARY)
                            mask_pred_j = mask_pred_j.astype('uint8')
                            mask_pred_j, _ = cv.findContours(mask_pred_j, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE,
                                                             offset=[int(x1.round()), int(y1.round())])
                            if not mask_pred_j:
                                continue
                            elif mask_pred_j[0].size <= 4:
                                continue
                            mask_pred_j = mask_pred_j[0][:, 0, :].astype('double').tolist()
                            mask_result = []
                            for point in mask_pred_j:
                                mask_result += point
                            """mask_pred_j = torch.tensor(mask_pred_j, dtype=torch.uint8)
                            mask_pred_j = F.pad(mask_pred_j,
                                                [int(x1.round()), int(w - x2.round()) - 1,
                                                 int(y1.round()), int(h - y2.round()) - 1],
                                                'constant', 0)
                            mask_pred_j = np.asfortranarray(mask_pred_j, dtype='uint8')
                            mask_pred_j = encode(mask_pred_j)"""
                            # get result
                            bbox_pred_i_j = list(bbox_pred_i_j)
                            for k in range(len(bbox_pred_i_j)):
                                bbox_pred_i_j[k] = float(bbox_pred_i_j[k])
                            result = {'image_id': img_id[0], 'score': float(prob_i_j), 'category_id': label_pred_i_j,
                                      'bbox': bbox_pred_i_j, 'segmentation': [mask_result]}
                            if img_id not in img_ids:
                                img_ids.append(img_id)
                            results.append(result)
    # 写入json文件
    json.dump(results, open(cfg.PATH_RESULTS, 'w'))
    # eval
    dataset.eval_coco(cfg.PATH_RESULTS, img_ids)
