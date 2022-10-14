# Medical Masked Autoencoders

## Paper
This repository provides the official implementation of training Vision Transformers (ViT) for (2D) medical imaging tasks as well as the usage of the pre-trained ViTs in the following paper:

<b>Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification</b> <br/>
[Junfei Xiao](https://lambert-x.github.io/), [Yutong Bai](https://scholar.google.com/citations?user=N1-l4GsAAAAJ&hl=en), [Alan Yuille](https://scholar.google.com/citations?user=FJ-huxgAAAAJ&hl=en&oi=ao), [Zongwei Zhou](https://www.zongweiz.com/) <br/>
Johns Hopkins University <br/>
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023 <br/>
[paper]() | [code]()

## Image reconstruction demo

<p align="center"><img src="figures/fig_reconstruction.png" width="100%"></p>

## Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in Table 1:

| architecture    | Pre-training dataset | Method | Checkpoint | 
|-----------------|:--------------------:|:------------------:|:-----------------:|
| DenseNet-121    | ImageNet (14M)       | Categorization  | download |
| ResNet-50       | ImageNet (14M)       | MoCo v2  | download |
| ResNet-50       | ImageNet (14M)       | BYOL  | download |
| ResNet-50       | ImageNet (14M)       | SwAV  | download |
| DenseNet-121    | X-rays (0.3M)        | MoCo v2  | download |
| DenseNet-121    | X-rays (0.3M)        | MAE  | download |
| ViT-Small/16    | ImageNet (14M)       | Categorization  | download |
| ViT-Small/16    | ImageNet (14M)       | MAE  | download |
| ViT-Small/16    | X-rays (0.3M)        | MAE  | download |
| ViT-Base/16     | X-rays (0.5M)        | MAE  | download |

## Acknowledgement
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research.
