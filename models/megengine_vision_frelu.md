---
template: hub1
title: FReLU 
summary:
    en_US: "Funnel Activation for Visual Recognition"
    zh_CN: FReLU （ImageNet 预训练权重）
author: MegEngine Team
tags: [vision, classification]
github-link: https://github.com/megvii-model/FunnelAct
---

```python
import megengine.hub
model = megengine.hub.load('megengine/models', 'resnet50_frelu', pretrained=True)
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

|        Model             | Activation | Top-1 err.|
| :----------------------  | :--------: | :------:  |
|    ResNet50              |  ReLU      | 24.0      |
|    ResNet50              |  PReLU     | 23.7      |
|    ResNet50              |  Swish     | 23.5      |
|    ResNet50              |  FReLU     | **22.4**  |
|    ShuffleNetV2 0.5x     |  ReLU      | 39.6      |
|    ShuffleNetV2 0.5x     |  PReLU     | 39.1      |
|    ShuffleNetV2 0.5x     |  Swish     | 38.7      |
|    ShuffleNetV2 0.5x     |  FReLU     | **37.1**  |

### 参考文献

- [Funnel Activation for Visual Recognition](https://arxiv.org/abs/2007.11824), Ma, Ningning, et al. "Funnel Activation for Visual Recognition." Proceedings of the European Conference on Computer Vision (ECCV). 2020.

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

|        Model             | Activation | Top-1 err.|
| :----------------------  | :--------: | :------:  |
|    ResNet50              |  ReLU      | 24.0      |
|    ResNet50              |  PReLU     | 23.7      |
|    ResNet50              |  Swish     | 23.5      |
|    ResNet50              |  FReLU     | **22.4**  |
|    ShuffleNetV2 0.5x     |  ReLU      | 39.6      |
|    ShuffleNetV2 0.5x     |  PReLU     | 39.1      |
|    ShuffleNetV2 0.5x     |  Swish     | 38.7      |
|    ShuffleNetV2 0.5x     |  FReLU     | **37.1**  |

### References

- [Funnel Activation for Visual Recognition](https://arxiv.org/abs/2007.11824), Ma, Ningning, et al. "Funnel Activation for Visual Recognition." Proceedings of the European Conference on Computer Vision (ECCV). 2020.
