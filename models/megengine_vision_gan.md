---
template: hub1
title: Generative Adversarial Networks
summary:
    en_US: GAN pre-trained on Cifar10
    zh_CN: 生产对抗网络（Cifar10 预训练权重）
author: MegEngine Team
tags: [vision, gan]
github-link: https://github.com/MegEngine/Models/tree/master/official/vision/gan
---

```python3
import megengine.hub as hub
import megengine_mimicry.nets.dcgan.dcgan_cifar as dcgan
import megengine_mimicry.utils.vis as vis

netG = dcgan.DCGANGeneratorCIFAR()
netG.load_state_dict(hub.load_serialized_obj_from_url("https://data.megengine.org.cn/models/weights/dcgan_cifar.pkl"))
images = dcgan_generator.generate_images(num_images=64)  # in NCHW format with normalized pixel values in [0, 1]
grid = vis.make_grid(images)  # in HW3 format with [0, 255] BGR images for visualization
vis.save_image(grid, "visual.png")
```

<!-- section: zh_CN -->

#### 训练参数
| 分辨率 | 批大小 | 学习率 | β<sub>1</sub> | β<sub>2</sub> | 衰减法则 | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 32 x 32 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 5 | 100K |


#### 评测指标
| Metric | Method |
|:--------------------------------:|:---------------------------------------:|
| [Inception Score (IS)](https://arxiv.org/abs/1606.03498) | 分成10份共计 50K 样本 |
| [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) | 50K 真实/生成样本 |
| [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) | 50K 真实/生成样本， 分成10份取平均值|


#### Cifar10 结果
| 模型 | FID Score | IS Score | KID Score |
| :-: | :-: | :-: | :-: |
| DCGAN  | 27.2 | 7.0 | 0.0242 |
| WGAN-WC  | 30.5  | 6.7 | 0.0249 |


### 参考文献

 - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), Alec Radford, Luke Metz, and Soumith Chintala.
 - [Wasserstein GAN](https://arxiv.org/abs/1701.07875), Martin Arjovsky, Soumith Chintala, and Léon Bottou.

<!-- section: en_US -->

#### Training Parameters
| Resolution | Batch Size | Learning Rate | β<sub>1</sub> | β<sub>2</sub> | Decay Policy | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 32 x 32 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 5 | 100K |


#### Metrics
| Metric | Method |
|:--------------------------------:|:---------------------------------------:|
| [Inception Score (IS)](https://arxiv.org/abs/1606.03498) | 50K samples at 10 splits|
| [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) | 50K real/generated samples |
| [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) | 50K real/generated samples, averaged over 10 splits.|


#### Cifar10 Results
| Method | FID Score | IS Score | KID Score |
| :-: | :-: | :-: | :-: |
| DCGAN  | 27.2 | 7.0 | 0.0242 |
| WGAN-WC  | 30.5  | 6.7 | 0.0249 |

### References

 - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), Alec Radford, Luke Metz, and Soumith Chintala.
 - [Wasserstein GAN](https://arxiv.org/abs/1701.07875), Martin Arjovsky, Soumith Chintala, and Léon Bottou.
