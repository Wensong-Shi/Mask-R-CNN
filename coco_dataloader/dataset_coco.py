import os
from PIL import Image

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class DatasetCoco(Dataset):
    def __init__(self, mode, dataset_type, dataset_root_dir, short_side, img_normalize_info):
        super().__init__()
        assert mode in ['TRAIN', 'EVAL'], 'Wrong mode!'
        self.mode = mode
        self.dataset_type = dataset_type
        self.dataset_root_dir = dataset_root_dir
        self.coco = COCO(self.get_ann_path())  # coco api
        self.short_side = short_side
        self.img_normalize_info = img_normalize_info
        # get categories and ids
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.clsnames = ['_background'] + [c['name'] for c in categories]
        self.num_cls = len(self.clsnames)
        self.img_ids = list(self.coco.imgToAnns.keys()) if self.dataset_type in ['train2017', 'val2017'] else self.coco.getImgIds()
        self.dict_catid2clsid = dict(zip(self.coco.getCatIds(), list(range(1, self.num_cls))))
        self.dict_clsid2catid = dict(zip(list(range(1, self.num_cls)), self.coco.getCatIds()))

    def __getitem__(self, index):
        if self.mode == 'TRAIN':
            img_id = self.img_ids[index]
            # get img
            img_path = self.imgid2imgpath(img_id)
            img = Image.open(img_path).convert('RGB')
            img, img_info = self.preprocess_img(img)
            # get ann
            labels, bboxes, masks = self.imgid2ann(img_id)
            # correct bboxes
            bboxes = bboxes / img_info[2]
            # correct masks
            masks = self.correct_mask(masks, img_info)
            return img_id, img, img_info, labels, bboxes, masks
        else:
            img_id = self.img_ids[index]
            # get img
            img_path = self.imgid2imgpath(img_id)
            img = Image.open(img_path).convert('RGB')
            img, img_info = self.preprocess_img(img)
            # don't need ann
            labels = torch.tensor([0])
            bboxes = torch.tensor([[0, 0, 0, 0]])
            masks = torch.tensor(np.zeros([1, int(img_info[0]), int(img_info[1])]))
            return img_id, img, img_info, labels, bboxes, masks

    def __len__(self):
        return len(self.img_ids)

    def get_ann_path(self):
        if self.dataset_type == 'train2017':
            return os.path.join(self.dataset_root_dir, 'annotations/instances_train2017.json')
        elif self.dataset_type == 'val2017':
            return os.path.join(self.dataset_root_dir, 'annotations/instances_val2017.json')
        elif self.dataset_type == 'test2017':
            return os.path.join(self.dataset_root_dir, 'annotations/image_info_test-dev2017.json')
        else:
            raise ValueError('Unsupport dataset type!')

    def imgid2imgpath(self, img_id):
        filename = self.coco.loadImgs(img_id)[0]['file_name']
        if self.dataset_type == 'train2017':
            img_path = os.path.join(self.dataset_root_dir, 'train2017', filename)
        elif self.dataset_type == 'val2017':
            img_path = os.path.join(self.dataset_root_dir, 'val2017', filename)
        elif self.dataset_type == 'test2017':
            img_path = os.path.join(self.dataset_root_dir, 'test2017', filename)
        else:
            raise ValueError('Unsupport dataset type!')
        assert os.path.exists(img_path), f'Image path does not exist: {img_path}'
        return img_path

    def preprocess_img(self, img):
        # scale img
        w_ori, h_ori = img.width, img.height
        if w_ori >= h_ori and h_ori < self.short_side:
            scale_factor = h_ori / self.short_side
            w = round(w_ori / scale_factor)
            h = round(h_ori / scale_factor)
        elif w_ori < h_ori and w_ori < self.short_side:
            scale_factor = w_ori / self.short_side
            w = round(w_ori / scale_factor)
            h = round(h_ori / scale_factor)
        else:
            scale_factor = 1
            w = round(w_ori / scale_factor)
            h = round(h_ori / scale_factor)
        # transform
        norm_mean = self.img_normalize_info['mean_rgb']
        norm_std = self.img_normalize_info['std_rgb']
        transform = transforms.Compose([transforms.Resize((h, w)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
        img = transform(img)
        img_info = torch.tensor([h, w, scale_factor], dtype=torch.float32)
        return img, img_info

    def imgid2ann(self, img_id):
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        labels = []
        bboxes = []
        masks = []
        for ann in anns:
            if ann['area'] <= 0:
                continue
            label = self.dict_catid2clsid[ann['category_id']]
            x1 = np.max([0, ann['bbox'][0]])
            y1 = np.max([0, ann['bbox'][1]])
            x2 = np.min([width - 1, x1 + np.max([0, ann['bbox'][2]]) - 1])
            y2 = np.min([height - 1, y1 + np.max([0, ann['bbox'][3]]) - 1])
            bbox = [x1, y1, x2, y2]
            mask = self.coco.annToMask(ann)
            labels.append(label)
            bboxes.append(bbox)
            masks.append(mask)
        labels = torch.tensor(labels, dtype=torch.uint8)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        # masks = torch.tensor(masks, dtype=torch.uint8)  暂不转换成tensor，便于后续处理
        return labels, bboxes, masks

    @staticmethod
    def correct_mask(masks_ori, img_info):
        masks = []
        for mask in masks_ori:
            mask = cv.resize(mask, [int(img_info[1]), int(img_info[0])], interpolation=cv.INTER_LINEAR)
            _, mask = cv.threshold(mask, 0.5, 1, cv.THRESH_BINARY)
            masks.append(mask)
        masks = np.array(masks)
        masks = torch.tensor(masks, dtype=torch.uint8)
        return masks

    # get mAP
    def eval_coco(self, path_results, img_ids):
        results_coco = self.coco.loadRes(path_results)
        coco_eval = COCOeval(self.coco, results_coco)
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
