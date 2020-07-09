---
template: hub1
title: ResNet
summary:
    en_US: Deep residual networks pre-trained on ImageNet
    zh_CN: 深度残差网络（ImageNet 预训练权重）
author: MegEngine Team
tags: [vision, classification]
github-link: https://github.com/megengine/models
---

```python
import megengine.hub
model = megengine.hub.load('megengine/models', 'resnet18', pretrained=True)
# or any of these variants
# model = megengine.hub.load('megengine/models', 'resnet34', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet50', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet101', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet152', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnext50_32x4d', pretrained=True)
model.eval()
```
<!-- section: zh_CN --> 

所有预训练模型希望数据被正确预处理。
模型要求输入BGR的图片, 短边缩放到`256`, 并中心裁剪至`(224 x 224)`的大小，最后做归一化处理 (均值为: `[103.530, 116.280, 123.675]`, 标准差为: `[57.375, 57.120, 58.395]`)。

下面是一段处理一张图片的样例代码。

```python
# Download an example image from the megengine data website
import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Read and pre-process the image
import cv2
import numpy as np
import megengine.data.transform as T
import megengine.functional as F

image = cv2.imread("cat.jpg")
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),  # BGR
    T.ToMode("CHW"),
])
processed_img = transform.apply(image)[np.newaxis, :]  # CHW -> 1CHW
logits = model(processed_img)
probs = F.softmax(logits)
print(probs)
```

### 模型描述

目前我们提供了以下几个预训练模型，分别是`resnet18`, `resnet34`, `resnet50`, `resnet101`，`resnet152`, `resnext50_32x4d`，它们在ImageNet验证集上的单crop性能如下表：

| 模型 | Top1 acc | Top5 acc |
| --- | --- | --- |
| ResNet18 |  70.312  |  89.430  | 
| ResNet34 |  73.960  |  91.630  | 
| ResNet50 |  76.254  |  93.056  | 
| ResNet101|  77.944  |  93.844  | 
| ResNet152|  78.582  |  94.130  |
| ResNeXt50 32x4d | 77.592 | 93.644 |

### 参考文献

 - [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778
 - [Aggregated Residual Transformation for Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf), Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1492-1500

<!-- section: en_US --> 

All pre-trained models expect input images normalized in the same way,
i.e. input images must be 3-channel BGR images of shape `(H x W x 3)`, and reszied shortedge to `256`, center-cropped to `(224 x 224)`.
The images should be normalized using `mean = [103.530, 116.280, 123.675]` and `std = [57.375, 57.120, 58.395])`.

Here's a sample execution.

```python
# Download an example image from the megengine data website
import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Read and pre-process the image
import cv2
import numpy as np
import megengine.data.transform as T
import megengine.functional as F

image = cv2.imread("cat.jpg")
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),  # BGR
    T.ToMode("CHW"),
])
processed_img = transform.apply(image)[np.newaxis, :]  # CHW -> 1CHW
logits = model(processed_img)
probs = F.softmax(logits)
print(probs)
```

### Model Description

Currently we provide these pretrained models: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`. Their 1-crop accuracy on ImageNet validation dataset can be found in following table.

| model | Top1 acc | Top5 acc |
| --- | --- | --- |
| ResNet18 |  70.312  |  89.430  | 
| ResNet34 |  73.960  |  91.630  | 
| ResNet50 |  76.254  |  93.056  | 
| ResNet101|  77.944  |  93.844  | 
| ResNet152|  78.582  |  94.130  |
| ResNeXt50 32x4d | 77.592 | 93.644 |

### References

 - [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778
 - [Aggregated Residual Transformation for Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf), Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1492-1500


