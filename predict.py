from PIL import Image
from random import randint

import numpy as np
import cv2 as cv
import torch
from torchvision.ops import nms

import cfg
from coco_dataloader import dataset_coco
from models.mask_rcnn import MaskRCNN


device = 'cuda' if torch.cuda.is_available() else 'cpu'


mode = ['TRAIN', 'EVAL'][1]
dataset_type = ['train2017', 'val2017', 'test2017'][1]
path_checkpoints = 'train_backup/test.pth'
img_path = 'coco/test2017/000000000001.jpg'

# prepare base things
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# prepare img
dataset = dataset_coco.DatasetCoco(mode, dataset_type, cfg.DATASET_ROOT_DIR, cfg.SHORT_SIDE, cfg.IMAGE_NORMALIZE_INFO)
img = Image.open(img_path).convert('RGB')
img, img_info = dataset.preprocess_img(img)
gt_labels = [torch.tensor([0])]
gt_bboxes = [torch.tensor([[0, 0, 0, 0]])]
gt_masks = [torch.tensor(np.zeros([1, int(img_info[0]), int(img_info[1])]))]
img = torch.stack([img])
img_info = torch.stack([img_info])
# prepare model
model = MaskRCNN(mode)
if use_cuda:
    model.cuda()
checkpoints = torch.load(path_checkpoints)
model.load_state_dict(checkpoints)
print(f'Load weights from {path_checkpoints}.')
# predict
model.eval()
with torch.no_grad():
    if use_cuda:
        for i in range(len(gt_bboxes)):
            gt_labels[i] = gt_labels[i].cuda()
            gt_bboxes[i] = gt_bboxes[i].cuda()
            gt_masks[i] = gt_masks[i].cuda()
    prob, labels_pred, bboxes_pred, masks_pred = model(img.type(FloatTensor), img_info, gt_labels, gt_bboxes, gt_masks)
    # process results
    index_bboxes = torch.arange(len(prob), device=device)
    _, index_mask2bbox = torch.sort(prob, descending=True)
    index_mask2bbox = index_mask2bbox[:100]
    img = cv.imread(img_path)
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
                # draw a result
                for prob_i_j, label_pred_i_j, bbox_pred_i_j, index_bbox_i_j in zip(prob_i, labels_pred_i, bboxes_pred_i,
                                                                                   index_bboxes_i):
                    if index_bbox_i_j in index_mask2bbox:
                        _, _, scale_factor = img_info[0]
                        mask_pred_j = masks_pred[index_mask2bbox == index_bbox_i_j][0]
                        # get class name
                        clsname = dataset.clsnames[int(label_pred_i_j)]
                        # get bbox result
                        bbox_pred_i_j *= scale_factor
                        # get mask result
                        w = (bbox_pred_i_j[2] - bbox_pred_i_j[0] + 1).ceil()
                        h = (bbox_pred_i_j[3] - bbox_pred_i_j[1] + 1).ceil()
                        mask_pred_j = np.array(mask_pred_j.cpu())
                        mask_pred_j = cv.resize(mask_pred_j, [int(w), int(h)], interpolation=cv.INTER_LINEAR)
                        _, mask_pred_j = cv.threshold(mask_pred_j, 0.5, 1, cv.THRESH_BINARY)
                        mask_pred_j = mask_pred_j.astype('uint8')
                        # draw a result
                        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
                        # draw a label
                        cv.putText(img, clsname + ': {0:.2f}'.format(float(prob_i_j)),
                                   [int(bbox_pred_i_j[0].round()), int(bbox_pred_i_j[1].round()) - 5],
                                   cv.FONT_HERSHEY_DUPLEX, 0.5, color)
                        # draw a bbox
                        cv.rectangle(img,
                                     [int(bbox_pred_i_j[0].round()), int(bbox_pred_i_j[1].round())],
                                     [int(bbox_pred_i_j[2].round()), int(bbox_pred_i_j[3].round())], color)
                        # draw a mask
                        mask_pred_j, _ = cv.findContours(mask_pred_j, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE,
                                                         offset=[int(bbox_pred_i_j[0].round()),
                                                                 int(bbox_pred_i_j[1].round())])
                        zeros = np.zeros(img.shape, dtype='uint8')
                        mask_pred_j = cv.fillPoly(zeros, mask_pred_j, color)
                        img = (0.3 * mask_pred_j + img).astype('uint8')
    # save
    cv.imwrite('predict.jpg', img)
