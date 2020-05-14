---
template: hub1
title: SimpleBaseline
summary:
    en_US: SimpleBaeline on COCO
    zh_CN: SimpleBaeline（COCO 预训练权重）
author: MegEngine Team
tags: [vision, keypoints]
github-link: https://github.com/megengine/models
---

```python3
import megengine.hub
model = megengine.hub.load('megengine/models', 'SimpleBaseline_Res50', pretrained=True)
# or any of these variants
# model = megengine.hub.load('megengine/models', 'SimpleBaseline_Res101', pretrained=True)
# model = megengine.hub.load('megengine/models', 'SimpleBaseline_Res152', pretrained=True)
model.eval()
```
<!-- section: zh_CN --> 

中文文档

<!-- section: en_US --> 

English Doc
