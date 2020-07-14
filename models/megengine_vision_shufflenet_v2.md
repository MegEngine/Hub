---
template: hub1
title: ShuffleNet V2
summary:
    en_US: "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    zh_CN: ShuffleNet V2（ImageNet 预训练权重）
author: MegEngine Team
tags: [vision, classification]
github-link: https://github.com/MegEngine/Models/tree/master/official/vision/classification
---

```python
import megengine.hub
model = megengine.hub.load('megengine/models', 'shufflenet_v2_x1_0', pretrained=True)
# model = megengine.hub.load('megengine/models', 'shufflenet_v2_x0_5', pretrained=True)
# model = megengine.hub.load('megengine/models', 'shufflenet_v2_x1_5', pretrained=True)
# model = megengine.hub.load('megengine/models', 'shufflenet_v2_x2_0', pretrained=True)
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

目前我们提供了部分在ImageNet上的预训练模型(见下表)，各个网络结构在ImageNet验证集上的表现如下：

| 模型 | top1 acc | top5 acc |
| --- | --- | --- |
| ShuffleNetV2 x0.5 |  60.696  |  82.190  |
| ShuffleNetV2 x1.0 |  69.372  |  88.764  |
| ShuffleNetV2 x1.5 |  72.806  |  90.792  |
| ShuffleNetV2 x2.0 |  75.074  |  92.278  |

### 参考文献

- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164), Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

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

Currently we provide several pretrained models(see the table below), Their 1-crop accuracy on ImageNet validation dataset can be found in following table.

| model | top1 acc | top5 acc |
| --- | --- | --- |
| ShuffleNetV2 x0.5 |  60.696  |  82.190  |
| ShuffleNetV2 x1.0 |  69.372  |  88.764  |
| ShuffleNetV2 x1.5 |  72.806  |  90.792  |
| ShuffleNetV2 x2.0 |  75.074  |  92.278  |

### References

 - [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164), Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
