# Mask-R-CNN
一个Mask R-CNN的Pytorch实现，使用COCO2017数据集训练。
## 介绍
这是一个遵循原文的Mask R-CNN的Pytorch实现，原论文：[Mask R-CNN; Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick; Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2961-2969](https://arxiv.org/abs/1703.06870)  
所用数据集为COCO2017数据集，backbone为ResNet-101-FPN。
## 环境
系统：Ubuntu20.04  
Python：3.8  
opencv-python：4.6.0.66  
torch：1.13.0  
torchvision：0.14.0
## 训练
### 1
自行下载COCO2017数据集并将图片文件与注释文件放在'coco'文件夹下。
### 2
在'coco/PythonAPI'文件夹下运行`make`
### 3
在'cfg.py'与'train.py'文件中适当修改相应超参数与设置。
### 4
运行'train.py'。
#### 初次训练时可能需要下载ResNet的预训练权重，注意联网。
## 评价
### 1
与训练类似，自行准备数据集。
### 2
若已执行过“训练”中的步骤2，则可跳过，否则需执行。
### 3
在'cfg.py'与'eval.py'文件中适当修改相应超参数与设置。
### 4
运行'eval.py'。
## 预测
### 1
准备一张需要预测的图片。
### 2
若已执行过“训练”中的步骤2，则可跳过，否则需执行。
### 3
在'cfg.py'与'predict.py'文件中适当修改相应超参数与设置。
### 4
运行'predict.py'。
## 注意
此仓库暂不支持多GPU训练。
