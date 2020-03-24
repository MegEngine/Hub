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

```python3
import megengine.hub
model = megengine.hub.load('megengine/models', 'resnet18', pretrained=True)
# or any of these variants
# model = megengine.hub.load('megengine/models', 'resnet34', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet50', pretrained=True)
# model = megengine.hub.load('megengine/models', 'resnet101', pretrained=True)
model.eval()
```
<!-- section: zh_CN --> 

中文文档

<!-- section: en_US --> 

English Doc
