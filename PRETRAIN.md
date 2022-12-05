# Pre-training

## Fully-supervised with ImageNet (CNNs & ViTs)

Please refer to [TorchVision Offical](https://pytorch.org/vision/stable/models.html) for CNN models and [DeiT: Data-efficient Image Transformers](https://github.com/facebookresearch/deit/blob/main/README_deit.md) for ViTs. The used checkpoints are provided in this [Table](https://github.com/lambert-x/medical_mae/blob/main/README.md#fine-tuning-with-pre-trained-checkpoints).

## Self-supervised with ImageNet (CNNs & ViTs)

For the CNN-based methods (MoCo V2, BYOL and SwAV), please refer to the official repositories [MoCo Official](https://github.com/facebookresearch/moco), [SwAV Official](https://github.com/facebookresearch/swav), and  [BYOL Official](https://github.com/deepmind/deepmind-research/tree/master/byol) for pre-training details. The used checkpoints are provided in this [Table](https://github.com/lambert-x/medical_mae/blob/main/README.md#fine-tuning-with-pre-trained-checkpoints).

For MAE pretraining for ViT on ImageNet, please refer to the [MAE Official](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md).

## Self-supervised with Chest X-rays (CNNs)

For MoCo V2, we use the code from [MoCo Official](https://github.com/facebookresearch/moco) and just replace the dataloader with the one provided in this repo.

For MAE, we implement [a version](https://github.com/lambert-x/medical_mae/blob/15a984a69d48b94563fd34a709c52eb7f9e46a55/models_mae_cnn.py#L27) based on the U-Net framework using [Segmentation models with pretrained backbones. PyTorch](https://github.com/qubvel/segmentation_models.pytorch). The pretraining task is the reconstruction task similar to the original MAE.

###### The pretraining command on a 8-GPU machine is shown as below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray_cnn.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 256 \
 --model 'densenet121' \
 --mask_ratio 0.75 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --random_resize_range 0.5 1.0 \
 --datasets_names chexpert chestxray_nih
```

## Self-supervised with Chest X-rays (ViTs)

We pretrain ViTs with MAE following the official repo but with **a customized recipe **(please refer to the paper for more details). Two sample commands are provided below.

###### To pretrain ViT-S on CheXpert and NIH Chest-Xray:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 256 \
 --model mae_vit_small_patch16_dec512d2b \
 --mask_ratio 0.90 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'random' \
 --random_resize_range 0.5 1.0 \
 --datasets_names chexpert chestxray_nih
```

###### To pretrain ViT-B on CheXpert, NIH Chest-Xray and MIMIC:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
 --use_env main_pretrain_multi_datasets_xray.py \
 --output_dir ${SAVE_DIR} \
 --log_dir ${SAVE_DIR} \
 --batch_size 256 \
 --model mae_vit_base_patch16_dec512d8b \
 --mask_ratio 0.90 \
 --epochs 800 \
 --warmup_epochs 40 \
 --blr 1.5e-4 --weight_decay 0.05 \
 --num_workers 8 \
 --input_size 224 \
 --mask_strategy 'random' \
 --random_resize_range 0.5 1.0 \
 --datasets_names chexpert chestxray_nih mimic_cxr
```
