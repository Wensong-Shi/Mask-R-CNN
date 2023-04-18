import os
import logging

import torch
from torch.utils.data import DataLoader

import cfg
from utils.misc import adjust_lr
from coco_dataloader import dataset_coco, collate_fn_coco
from models.mask_rcnn import MaskRCNN


mode = ['TRAIN', 'EVAL'][0]
dataset_type = ['train2017', 'val2017', 'test2017'][1]

# prepare base things
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(cfg.LOGFILE_TRAIN), logging.StreamHandler()])
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
epoch_start = 1
epoch_end = cfg.MAX_EPOCH
# prepare dataloader
dataset = dataset_coco.DatasetCoco(mode, dataset_type, cfg.DATASET_ROOT_DIR, cfg.SHORT_SIDE, cfg.IMAGE_NORMALIZE_INFO)
dataloader = DataLoader(dataset=dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                        collate_fn=collate_fn_coco.collate_fn_coco, pin_memory=cfg.PIN_MEMORY)
# prepare model
model = MaskRCNN(mode)
if use_cuda:
    model.cuda()
# prepare optimizer
index_lr = 0
if cfg.IS_WARMUP:
    lr = cfg.LR[index_lr] / 10
else:
    lr = cfg.LR[index_lr]
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
# train
logging.info('Start training!')
model.set_train()
for epoch in range(epoch_start, epoch_end + 1):
    logging.info(f'Start epoch {epoch}.')
    # adjust lr
    if epoch == cfg.LR_ADJUST_EPOCH:
        index_lr += 1
        adjust_lr(optimizer, cfg.LR[index_lr])
        logging.info(f'Adjust lr to {cfg.LR[index_lr]}.')
    # train one epoch
    for batch_index, data in enumerate(dataloader):
        # warm up
        if epoch == 1 and cfg.IS_WARMUP and (batch_index < cfg.NUM_WARMUP_ITERS):
            assert index_lr == 0, 'There are some bugs...'
            lr = cfg.LR[index_lr] / 10
            lr += (cfg.LR[index_lr] - cfg.LR[index_lr] / 10) * batch_index / (cfg.NUM_WARMUP_ITERS - 1)
            adjust_lr(optimizer, lr)
        # optimize
        optimizer.zero_grad()
        _, img, img_info, gt_labels, gt_bboxes, gt_masks = data
        if use_cuda:
            for i in range(len(gt_bboxes)):
                gt_labels[i] = gt_labels[i].cuda()
                gt_bboxes[i] = gt_bboxes[i].cuda()
                gt_masks[i] = gt_masks[i].cuda()
        loss = model(img.type(FloatTensor), img_info, gt_labels, gt_bboxes, gt_masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_NORM_GRAD_CLIP)
        optimizer.step()
        logging.info(f'Epoch: {epoch}/{epoch_end} Batch: {batch_index + 1}/{len(dataloader)} Loss: {loss}')
    # save model
    if epoch > cfg.LR_ADJUST_EPOCH:
        path_savefile = os.path.join(cfg.BACKUP_DIR_TRAIN, f'epoch{epoch}_loss{loss}.pth')
        logging.info('Saving model...')
        torch.save(model.state_dict(), path_savefile)
