---
template: hub1
title: DeepLabV3plus
summary:
    en_US: DeepLabV3plus pre-trained on VOC
    zh_CN: DeepLabV3plus (VOC预训练权重）
author: MegEngine Team
tags: [vision]
github-link: https://github.com/megengine/models
---

```python
from megengine import hub
model = hub.load(
    "megengine/models",
    "deeplabv3plus_res101",
    pretrained=True,
)
model.eval()
```
<!-- section: zh_CN --> 

所有预训练模型希望数据被正确预处理。模型要求输入BGR的图片, 建议缩放到512x512，最后做归一化处理 (均值为: `[103.530, 116.280, 123.675]`, 标准差为: `[57.375, 57.120, 58.395]`)。


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

import megengine.jit as jit
@jit.trace(symbolic=True, opt_level=2)
def pred_fun(data, net=None):
    net.eval()
    pred = net(data)
    return pred

image = cv2.imread("cat.jpg")
orih, oriw = image.shape[:2]
transform = T.Compose([
    T.Resize((512, 512)),
    T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),  # BGR
    T.ToMode(),
])
processed_img = transform.apply(image)[np.newaxis]  # CHW -> 1CHW
pred = pred_fun(processed_img, net=model)

pred = pred.numpy().squeeze().argmax(axis=0)
pred = cv2.resize(pred.astype("uint8"), (oriw, orih), interpolation=cv2.INTER_LINEAR)
```

### 模型描述

目前我们提供了 deeplabv3plus 的预训练模型, 在voc验证集的表现如下：

 Methods     | Backbone    | TrainSet  | EvalSet | mIoU_single   | mIoU_multi  |
 :--:        |:--:         |:--:       |:--:     |:--:           |:--:         |
 DeepLab v3+ | ResNet101   | train_aug | val     | 79.0          | 79.8        |

### 参考文献

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611.pdf), Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and
Hartwig Adam; ECCV, 2018

<!-- section: en_US --> 

All pre-trained models expect input images normalized in the same way. Input images must be 3-channel BGR images of shape (H x W x 3), reszied to (512 x 512), then normalized using mean = [103.530, 116.280, 123.675] and std = [57.375, 57.120, 58.395]).

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

import megengine.jit as jit
@jit.trace(symbolic=True, opt_level=2)
def pred_fun(data, net=None):
    net.eval()
    pred = net(data)
    return pred

image = cv2.imread("cat.jpg")
orih, oriw = image.shape[:2]
transform = T.Compose([
    T.Resize((512, 512)),
    T.Normalize(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]),  # BGR
    T.ToMode(),
])
processed_img = transform.apply(image)[np.newaxis, :]  # CHW -> 1CHW
pred = pred_fun(processed_img, net=model)

pred = pred.numpy().squeeze().argmax(axis=0)
pred = cv2.resize(pred.astype("uint8"), (oriw, orih), interpolation=cv2.INTER_LINEAR)
```

### Model Description

 Methods     | Backbone    | TrainSet  | EvalSet | mIoU_single   | mIoU_multi  |
 :--:        |:--:         |:--:       |:--:     |:--:           |:--:         |
 DeepLab v3+ | ResNet101   | train_aug | val     | 79.0          | 79.8        |

### References

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611.pdf), Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and
Hartwig Adam; ECCV, 2018
