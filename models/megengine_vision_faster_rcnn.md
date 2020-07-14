---
template: hub1
title: Faster-RCNN
summary:
    en_US: Faster-RCNN pre-trained on COCO
    zh_CN: Faster-RCNN (COCO预训练权重）
author: MegEngine Team
tags: [vision, detection]
github-link: https://github.com/megengine/models
---

```python
from megengine import hub
model = hub.load(
    "megengine/models",
    "faster_rcnn_res50_coco_1x_800size",
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
image = cv2.imread("cat.jpg")

data, im_info = models_api.DetEvaluator.process_inputs(image, 800, 1333)
model.inputs["image"].set_value(data)
model.inputs["im_info"].set_value(im_info)

from megengine import jit
@jit.trace(symbolic=True)
def infer():
    predictions = model(model.inputs)
    return predictions

print(infer())
```

### 模型描述

目前我们提供了retinanet的预训练模型, 在coco验证集上的结果如下：

| model                               | mAP<br>@5-95 |
| ---                                 | :---:        |
| faster-rcnn-res50-1x-800size        | 38.8         |
| faster-rcnn-res50-1x-800size-syncbn | 39.3         |

### 参考文献

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) S. Ren, K. He, R. Girshick, and J. Sun. In: Neural Information Processing Systems(NIPS)(2015).
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf) T. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan and S. Belongie. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 936-944, doi: 10.1109/CVPR.2017.106.
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf)  Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence, Lin T Y, Maire M, Belongie S, et al. European conference on computer vision. Springer, Cham, 2014: 740-755.


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
image = cv2.imread("cat.jpg")

data, im_info = models_api.DetEvaluator.process_inputs(image, 800, 1333)
model.inputs["image"].set_value(data)
model.inputs["im_info"].set_value(im_info)

from megengine import jit
@jit.trace(symbolic=True)
def infer():
    predictions = model(model.inputs)
    return predictions

print(infer())
```

### Model Description

Currently we provide a `retinanet` model which is pretrained on `COCO2017` training set. The mAP on `COCO2017` val set can be found in following table.

| model                               | mAP<br>@5-95 |
| ---                                 | :---:        |
| faster-rcnn-res50-1x-800size        | 38.8         |
| faster-rcnn-res50-1x-800size-syncbn | 39.3         |

### References

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf) S. Ren, K. He, R. Girshick, and J. Sun. In: Neural Information Processing Systems(NIPS)(2015).
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf) T. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan and S. Belongie. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 936-944, doi: 10.1109/CVPR.2017.106.
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf)  Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence, Lin T Y, Maire M, Belongie S, et al. European conference on computer vision. Springer, Cham, 2014: 740-755.
