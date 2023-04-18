import torch


def reg2bbox(bboxes_ori, reg):
    width = bboxes_ori[:, 2] - bboxes_ori[:, 0] + 1
    height = bboxes_ori[:, 3] - bboxes_ori[:, 1] + 1
    cx = bboxes_ori[:, 0] + width / 2
    cy = bboxes_ori[:, 1] + height / 2
    dx = reg[:, 0]
    dy = reg[:, 1]
    dw = reg[:, 2]
    dh = reg[:, 3]
    cx = width * dx + cx
    cy = height * dy + cy
    width = width * torch.exp(dw)
    height = height * torch.exp(dh)
    x1 = cx - width / 2
    y1 = cy - height / 2
    x2 = x1 + width - 1
    y2 = y1 + height - 1
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return bboxes  # 3*w*h * 4, [[x1, y1, x2, y2], ...], torch.float32


def clip_bbox(prob, bboxes, img_info):
    # 在collate_fn中对图像进行了填充，因此舍弃一些超出原图像范围的bboxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    index_x = x1 >= (img_info[1] - 1)
    index_y = y1 >= (img_info[0] - 1)
    index = index_x + index_y
    prob[index] = 1e-5
    # clip bbox
    index = ~index
    bboxes[index, 0] = bboxes[index, 0].clamp(0, img_info[1] - 1)
    bboxes[index, 1] = bboxes[index, 1].clamp(0, img_info[0] - 1)
    bboxes[index, 2] = bboxes[index, 2].clamp(0, img_info[1] - 1)
    bboxes[index, 3] = bboxes[index, 3].clamp(0, img_info[0] - 1)
    """
    prob: 3*w*h, [prob1, prob2, ...], torch.float32
    bboxes: 3*w*h * 4, [[x1, y1, x2, y2], ...], torch.float32
    """
    return prob, bboxes


def bbox2reg(bboxes_pred, gt_bboxes):
    width_pred = bboxes_pred[:, 2] - bboxes_pred[:, 0] + 1
    height_pred = bboxes_pred[:, 3] - bboxes_pred[:, 1] + 1
    cx_pred = bboxes_pred[:, 0] + width_pred / 2
    cy_pred = bboxes_pred[:, 1] + height_pred / 2
    width_gt = gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1
    height_gt = gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1
    cx_gt = gt_bboxes[:, 0] + width_gt / 2
    cy_gt = gt_bboxes[:, 1] + height_gt / 2
    # unmap
    dx = (cx_gt - cx_pred) / width_pred
    dy = (cy_gt - cy_pred) / height_pred
    dw = torch.log(width_gt / width_pred)
    dh = torch.log(height_gt / height_pred)
    reg = torch.stack([dx, dy, dw, dh], dim=-1)
    return reg  # num_positive * 4, [[dx, dy, dw, dh], ...], torch.float32


def intersection(proposals, gt_bboxes):
    x1 = torch.maximum(proposals[:, 0], gt_bboxes[:, 0])
    y1 = torch.maximum(proposals[:, 1], gt_bboxes[:, 1])
    x2 = torch.minimum(proposals[:, 2], gt_bboxes[:, 2])
    y2 = torch.minimum(proposals[:, 3], gt_bboxes[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)  # num_positive * 4, [[x1, y1, x2, y2], ...], torch.float32
