---
template: hub1
title: WeightNet 
summary:
    en_US: "WeightNet: Revisiting the Design Space of Weight Network"
    zh_CN: WeightNet - ShuffleNet V2（ImageNet 预训练权重）
author: MegEngine Team
tags: [vision, classification]
github-link: https://github.com/MegEngine/Models/tree/master/official/vision/classification
---

```python
import megengine.hub
model = megengine.hub.load('megengine/models', 'shufflenet_v2_x0_5', pretrained=True)
# model = megengine.hub.load('megengine/models', 'shufflenet_v2_x1_0', pretrained=True)
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


| Model               | #Params. | FLOPs | Top-1 err. |
|---------------------|----------|-------|------------|
| ShuffleNetV2 (0.5×) | 1.4M     | 41M   | 39.7       |
| + WeightNet (1×)    | 1.5M     | 41M   | **36.7**   |
| ShuffleNetV2 (1.0×) | 2.2M     | 138M  | 30.9       |
| + WeightNet (1×)    | 2.4M     | 139M  | **28.8**   |
| ShuffleNetV2 (1.5×) | 3.5M     | 299M  | 27.4       |
| + WeightNet (1×)    | 3.9M     | 301M  | **25.6**   |
| ShuffleNetV2 (2.0×) | 5.5M     | 557M  | 25.5       |
| + WeightNet (1×)    | 6.1M     | 562M  | **24.1**   |



| Model               | #Params. | FLOPs | Top-1 err. |
|---------------------|----------|-------|------------|
| ShuffleNetV2 (0.5×) | 1.4M     | 41M   | 39.7       |
| + WeightNet (8×)    | 2.7M     | 42M   | **34.0**   |
| ShuffleNetV2 (1.0×) | 2.2M     | 138M  | 30.9       |
| + WeightNet (4×)    | 5.1M     | 141M  | **27.6**   |
| ShuffleNetV2 (1.5×) | 3.5M     | 299M  | 27.4       |
| + WeightNet (4×)    | 9.6M     | 307M  | **25.0**   |
| ShuffleNetV2 (2.0×) | 5.5M     | 557M  | 25.5       |
| + WeightNet (4×)    | 18.1M    | 573M  | **23.5**   |

### 参考文献

- [WeightNet: Revisiting the Design Space of Weight Network](https://arxiv.org/abs/2007.11823), Ma, Ningning, et al. "WeightNet: Revisiting the Design Space of Weight Network." Proceedings of the European Conference on Computer Vision (ECCV). 2020.

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

| Model               | #Params. | FLOPs | Top-1 err. |
|---------------------|----------|-------|------------|
| ShuffleNetV2 (0.5×) | 1.4M     | 41M   | 39.7       |
| + WeightNet (1×)    | 1.5M     | 41M   | **36.7**   |
| ShuffleNetV2 (1.0×) | 2.2M     | 138M  | 30.9       |
| + WeightNet (1×)    | 2.4M     | 139M  | **28.8**   |
| ShuffleNetV2 (1.5×) | 3.5M     | 299M  | 27.4       |
| + WeightNet (1×)    | 3.9M     | 301M  | **25.6**   |
| ShuffleNetV2 (2.0×) | 5.5M     | 557M  | 25.5       |
| + WeightNet (1×)    | 6.1M     | 562M  | **24.1**   |



| Model               | #Params. | FLOPs | Top-1 err. |
|---------------------|----------|-------|------------|
| ShuffleNetV2 (0.5×) | 1.4M     | 41M   | 39.7       |
| + WeightNet (8×)    | 2.7M     | 42M   | **34.0**   |
| ShuffleNetV2 (1.0×) | 2.2M     | 138M  | 30.9       |
| + WeightNet (4×)    | 5.1M     | 141M  | **27.6**   |
| ShuffleNetV2 (1.5×) | 3.5M     | 299M  | 27.4       |
| + WeightNet (4×)    | 9.6M     | 307M  | **25.0**   |
| ShuffleNetV2 (2.0×) | 5.5M     | 557M  | 25.5       |
| + WeightNet (4×)    | 18.1M    | 573M  | **23.5**   |

### References

- [WeightNet: Revisiting the Design Space of Weight Network](https://arxiv.org/abs/2007.11823), Ma, Ningning, et al. "WeightNet: Revisiting the Design Space of Weight Network." Proceedings of the European Conference on Computer Vision (ECCV). 2020.
