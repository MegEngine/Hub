---
template: hub1
title: FCOS
summary:
    en_US: FCOS pre-trained on COCO2017
    zh_CN: FCOS (COCO2017预训练权重）
author: MegEngine Team
tags: [vision, detection]
github-link: https://github.com/MegEngine/Models/tree/master/official/vision/detection
---

```python
from megengine import hub
model = hub.load(
    "megengine/models",
    "fcos_res50_coco_3x_800size",
    pretrained=True,
    use_cache=False,
)
model.eval()

models_api = hub.import_module(
    "megengine/models",
    git_host="github.com",
)
```
<!-- section: zh_CN -->

所有预训练模型希望数据被正确预处理。
模型要求输入BGR的图片, 同时需要等比例缩放到：短边和长边分别不超过800/1333
最后做归一化处理 (均值为: `[103.530, 116.280, 123.675]`, 标准差为: `[57.375, 57.120, 58.395]`)。

下面是一段处理一张图片的样例代码。

```python
# Download an example image from the megengine data website
import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Read and pre-process the image
import cv2
import megengine as mge

image = cv2.imread("cat.jpg")
data, im_info = models_api.DetEvaluator.process_inputs(image, 800, 1333)
predictions = model(image=mge.tensor(data), im_info=mge.tensor(im_info))
print(predictions)
```

### 模型描述

目前我们提供了在COCO2017数据集上预训练的FCOS模型, 性能如下：

| model                        | mAP<br>@5-95 |
| ---                          | :---:        |
| fcos-res18-coco-3x-800size   | 37.6         |
| fcos-res34-coco-3x-800size   | 41.0         |
| fcos-res50-coco-3x-800size   | 42.2         |
| fcos-res101-coco-3x-800size  | 44.3         |
| fcos-resx101-coco-2x-800size | 44.8         |

### 参考文献

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. IEEE International Conference on Computer Vision (ICCV), 2019.
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. European Conference on Computer Vision (ECCV), 2014.

<!-- section: en_US -->

All pre-trained models expect input images normalized in the same way,
i.e. input images must be 3-channel BGR images of shape `(H x W x 3)`, and reszied shortedge/longedge to no more than `800/1333`.
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
import megengine as mge

image = cv2.imread("cat.jpg")
data, im_info = models_api.DetEvaluator.process_inputs(image, 800, 1333)
predictions = model(image=mge.tensor(data), im_info=mge.tensor(im_info))
print(predictions)
```

### Model Description

Currently we provide FCOS models pretrained on COCO2017 dataset. The performance can be found in following table.

| model                        | mAP<br>@5-95 |
| ---                          | :---:        |
| fcos-res18-coco-3x-800size   | 37.6         |
| fcos-res34-coco-3x-800size   | 41.0         |
| fcos-res50-coco-3x-800size   | 42.2         |
| fcos-res101-coco-3x-800size  | 44.3         |
| fcos-resx101-coco-2x-800size | 44.8         |

### References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. IEEE International Conference on Computer Vision (ICCV), 2019.
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. European Conference on Computer Vision (ECCV), 2014.
