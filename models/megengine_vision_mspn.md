---
template: hub1
title: MSPN
summary:
    en_US: MSPN on COCO
    zh_CN: MSPN（COCO 预训练权重）
author: MegEngine Team
tags: [vision, keypoints]
github-link: https://github.com/MegEngine/Models/tree/master/official/vision/keypoints
---

```python3
import megengine.hub
model = megengine.hub.load('megengine/models', 'mspn_4stage', pretrained=True)
model.eval()
```
<!-- section: zh_CN -->
MSPN是单人关节点检测模型，在多人场景下需要配合人体检测器使用。详细的多人检测代码示例可以参考[inference.py](https://github.com/MegEngine/Models/blob/master/official/vision/keypoints/inference.py)。

针对单张图片，这里提供使用retinanet做人体检测，然后用MSPN检测关节点的示例:

```python3

import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Read and pre-process the image
import cv2
image = cv2.imread("cat.jpg")

import official.vision.detection.retinanet_res50_coco_1x_800size as Det
detector = Det.retinanet_res50_1x_800size(pretrained=True)

models_api = hub.import_module(
    "megengine/models",
    git_host="github.com",
)

@jit.trace(symbolic=True)
def det_func():
    pred = detector(detector.inputs)
    return pred

@jit.trace(symbolic=True)
def keypoint_func():
    pred = model.predict()
    return pred

evaluator = models_api.KeypointEvaluator(
    detector,
    det_func,
    model,
    keypoint_func
    )

print("Detecting Persons")
person_boxes = evaluator.detect_persons(image)

print("Detecting Keypoints")
all_keypoints = evaluator.predict(image, person_boxes)

print("Visualizing")
canvas = evaluator.vis_skeletons(image, all_keypoints)
cv2.imwrite("vis_skeleton.jpg", canvas)
```

### 模型描述
本目录使用了在COCO val2017上的Human AP为56.4的人体检测结果，最后在COCO val2017上人体关节点估计结果为

|Methods|Backbone|Input Size| AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|:---:|---|---|---|---|---|---|---|---|---|---|---|
| MSPN_4stage |MSPN|256x192| 0.752 | 0.900 | 0.819 | 0.716 | 0.825 | 0.819 | 0.943 | 0.875 | 0.770 | 0.887 |

### 参考文献
- [Rethinking on Multi-Stage Networks for Human Pose Estimation](https://arxiv.org/pdf/1901.00148.pdf) Wenbo Li1, Zhicheng Wang, Binyi Yin, Qixiang Peng, Yuming Du, Tianzi Xiao, Gang Yu, Hongtao Lu, Yichen Wei and Jian Sun

<!-- section: en_US -->
SimpleBaseline is classical network for single person pose estimation. It can also be applied to multi-person cases when combined with a human detector. The details of this pipline can be referred to [inference.py](https://github.com/MegEngine/Models/blob/master/official/vision/keypoints/inference.py).

For single image, here is a sample execution when SimpleBaseline is combined with retinanet

```python3

import urllib
url, filename = ("https://data.megengine.org.cn/images/cat.jpg", "cat.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Read and pre-process the image
import cv2
image = cv2.imread("cat.jpg")

import official.vision.detection.retinanet_res50_coco_1x_800size as Det
detector = Det.retinanet_res50_1x_800size(pretrained=True)

models_api = hub.import_module(
    "megengine/models",
    git_host="github.com",
)

@jit.trace(symbolic=True)
def det_func():
    pred = detector(detector.inputs)
    return pred

@jit.trace(symbolic=True)
def keypoint_func():
    pred = model.predict()
    return pred

evaluator = models_api.KeypointEvaluator(
    detector,
    det_func,
    model,
    keypoint_func
    )

print("Detecting Persons")
person_boxes = evaluator.detect_persons(image)

print("Detecting Keypoints")
all_keypoints = evaluator.predict(image, person_boxes)

print("Visualizing")
canvas = evaluator.vis_skeletons(image, all_keypoints)
cv2.imwrite("vis_skeleton.jpg", canvas)
```
### Model Desription

With the AP human detectoin results being 56.4 on COCO val2017 dataset, the performances of simplebline on COCO val2017 dataset is

|Methods|Backbone|Input Size| AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|:---:|---|---|---|---|---|---|---|---|---|---|---|
| MSPN_4stage |MSPN|256x192| 0.752 | 0.900 | 0.819 | 0.716 | 0.825 | 0.819 | 0.943 | 0.875 | 0.770 | 0.887 |

### References
- [Rethinking on Multi-Stage Networks for Human Pose Estimation](https://arxiv.org/pdf/1901.00148.pdf) Wenbo Li1, Zhicheng Wang, Binyi Yin, Qixiang Peng, Yuming Du, Tianzi Xiao, Gang Yu, Hongtao Lu, Yichen Wei and Jian Sun
