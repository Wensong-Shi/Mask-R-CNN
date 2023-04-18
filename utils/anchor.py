import torch


def generate_base_anchor(anchor_size, anchor_ratios):
    h_base, w_base = anchor_size, anchor_size
    h_list = []
    w_list = []
    for anchor_ratio in anchor_ratios:
        h_list.append(h_base * (anchor_ratio ** 0.5))
        w_list.append(w_base / (anchor_ratio ** 0.5))
    base_anchor = []
    for w, h in zip(w_list, h_list):
        base_anchor.append([-(w - 1) / 2, -(h - 1) / 2, (w - 1) / 2, (h - 1) / 2])
    base_anchor = torch.tensor(base_anchor, dtype=torch.float32)
    return base_anchor  # 3 * 4, [[x1, y1, x2, y2], ...], torch.float32


def generate_anchor(feature_shape, feature_stride, anchor_size, anchor_ratios):
    base_anchor = generate_base_anchor(anchor_size, anchor_ratios)
    h, w = feature_shape
    x_shift = torch.arange(0, w) * feature_stride
    y_shift = torch.arange(0, h) * feature_stride
    xx_shift = x_shift.repeat(len(y_shift))
    yy_shift = y_shift.view(-1, 1).repeat(1, len(x_shift)).view(-1)
    shifts = torch.stack([xx_shift, yy_shift, xx_shift, yy_shift], dim=-1)
    anchors = base_anchor[None, :, :] + shifts[:, None, :]
    # 3*w*h * 4, [[x1, y1, x2, y2], ...], torch.float32
    return anchors.view(-1, 4).cuda() if torch.cuda.is_available() else anchors.view(-1, 4)
