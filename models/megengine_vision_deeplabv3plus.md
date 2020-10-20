---
template: hub1
title: DeepLabV3plus
summary:
    en_US: DeepLabV3+ pre-trained on Pascal VOC2012 or Cityscapes
    zh_CN: DeepLabV3+ (Pascal VOC2012或Cityscapes预训练权重）
author: MegEngine Team
tags: [vision]
github-link: https://github.com/MegEngine/Models/tree/master/official/vision/segmentation
---

```python
from megengine import hub
model = hub.load(
    "megengine/models",
    "deeplabv3plus_res101_voc_512size",
    pretrained=True,
)
model.eval()
```
<!-- section: zh_CN -->

所有预训练模型希望数据被正确预处理。模型要求输入BGR的图片, 建议缩放到512x512（Pascal VOC2012）或1024x2048（Cityscapes），最后做归一化处理 (均值为: `[103.530, 116.280, 123.675]`, 标准差为: `[57.375, 57.120, 58.395]`)。


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
orih, oriw = image.shape[:2]
transform = T.Compose([
    T.Resize((512, 512)),
    T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),  # BGR
    T.ToMode(),
])
processed_img = transform.apply(image)[np.newaxis]  # CHW -> 1CHW
pred = model(processed_img)
pred = pred.numpy().squeeze().argmax(axis=0)
pred = cv2.resize(pred.astype("uint8"), (oriw, orih), interpolation=cv2.INTER_LINEAR)
```

### 模型描述

目前我们提供了在Pascal VOC2012数据集或Cityscapes数据集上预训练的DeepLabV3+模型, 性能如下：

| model                                   | mIoU |
| ---                                     | :--: |
| deeplabv3plus-res101-voc-512size        | 79.5 |
| deeplabv3plus-res101-cityscapes-768size | 78.5 |

### 参考文献

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. European Conference on Computer Vision (ECCV), 2018.

<!-- section: en_US -->

All pre-trained models expect input images normalized in the same way. Input images must be 3-channel BGR images of shape (H x W x 3), reszied to (512 x 512) for Pascal VOC2012 or (1024 x 2048) for Cityscapes, then normalized using mean = [103.530, 116.280, 123.675] and std = [57.375, 57.120, 58.395]).

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
orih, oriw = image.shape[:2]
transform = T.Compose([
    T.Resize((512, 512)),
    T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),  # BGR
    T.ToMode(),
])
processed_img = transform.apply(image)[np.newaxis]  # CHW -> 1CHW
pred = model(processed_img)
pred = pred.numpy().squeeze().argmax(axis=0)
pred = cv2.resize(pred.astype("uint8"), (oriw, orih), interpolation=cv2.INTER_LINEAR)
```

### Model Description

Currently we provide DeepLabV3+ models pretrained on Pascal VOC2012 dataset or Cityscapes dataset. The performance can be found in following table.

| model                                   | mIoU |
| ---                                     | :--: |
| deeplabv3plus-res101-voc-512size        | 79.5 |
| deeplabv3plus-res101-cityscapes-768size | 78.5 |

### References

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. European Conference on Computer Vision (ECCV), 2018.
