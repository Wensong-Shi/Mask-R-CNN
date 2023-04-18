import torch
import torch.nn.functional as F


def collate_fn_coco(data_batch):
    # data_batch: [[img_id, img, img_info, labels, bboxes, masks], ...]
    height_max = max([data[1].shape[1] for data in data_batch])
    width_max = max([data[1].shape[2] for data in data_batch])
    img_id_batch = []
    img_batch = []
    img_info_batch = []
    labels_batch = []
    bboxes_batch = []
    masks_batch = []
    for data in data_batch:
        img_id, img, img_info, labels, bboxes, masks = data
        img_id_batch.append(img_id)
        # pad img
        img = F.pad(img, [0, width_max - img.shape[2], 0, height_max - img.shape[1]], 'constant', 0)
        img_batch.append(img)
        img_info_batch.append(img_info)
        labels_batch.append(labels)
        bboxes_batch.append(bboxes)
        masks_batch.append(masks)
    img_batch = torch.stack(img_batch)
    img_info_batch = torch.stack(img_info_batch)
    """
    Note: B = batch_size, N = num_gt_bboxes, H'与W'代表每张图片的高与宽
    img_id_batch: B, [img_id1, img_id2, ...], list(int)
    img: B * C * H * W, torch.float32
    img_info_batch: B * 3, [[h, w, scale_factor], ...], torch.float32
    labels_batch: B * (N), [[label1, label2, ...], ...], list(torch.uint8)
    bboxes_batch: B * (N * 4), [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...], ...], list(torch.float32)
    masks_batch: B * (N * H' * W'), list(torch.uint8)
    """
    return img_id_batch, img_batch, img_info_batch, labels_batch, bboxes_batch, masks_batch
