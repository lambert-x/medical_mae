# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
import cv2
import numpy as np
import torch.nn.functional as F


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, heatmap=None,
                 mask_strategy='random', weight_range=[0.1, 1.0], heatmap_binary_threshold=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # if heatmap is not None:
        #     self.heatmap_weights = self.extract_patch_weights(heatmap, img_size, patch_size,
        #                                                       weight_min=weight_range[0], weight_max=weight_range[1],
        #                                                       heatmap_binary_threshold=heatmap_binary_threshold)
        #

        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_strategy = mask_strategy
        self.weight_range = weight_range
        self.heatmap_binary_threshold = heatmap_binary_threshold
        self.local_attention_mask = self.get_local_attention_mask(img_size, patch_size).cuda()

    # def extract_patch_weights(self, heatmap, img_size, patch_size, weight_min=0.1, weight_max=1.0,
    #                           heatmap_binary_threshold=None):
    #     heatmap = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_AREA)
    #     heatmap = heatmap[:, :, 0]  # only need one channel for mask
    #     heatmap = heatmap.astype(np.float32)
    #
    #     if heatmap_binary_threshold is not None:
    #         if heatmap_binary_threshold == 'mean':
    #             threshold = np.mean(heatmap)
    #         else:
    #             raise NotImplementedError
    #         heatmap = (heatmap > threshold).astype(np.float32)
    #
    #     heatmap = torch.tensor(heatmap)
    #     h = w = heatmap.shape[0] // patch_size
    #     heatmap = heatmap.reshape(h, patch_size, w, patch_size)
    #     heatmap = torch.einsum('hpwq->hwpq', heatmap)
    #     heatmap_weights = heatmap.reshape(h * w, patch_size ** 2).sum(dim=-1)
    #     print('**************************')
    #     print(weight_min, weight_max)
    #     print('**************************')
    #     heatmap_weights = (heatmap_weights / heatmap_weights.max() * (weight_max - weight_min) + weight_min)
    #     return heatmap_weights

    def get_local_attention_mask(self, img_size, patch_size):
        h = w = img_size // patch_size
        masks = []
        for i in range(h):
            for j in range(w):
                mask = torch.zeros(h, w)

                x_min = max(0, i - 1)
                x_max = min(h - 1, i + 1)
                y_min = max(0, j - 1)
                y_max = min(w - 1, j + 1)

                mask[x_min:x_max + 1, y_min:y_max + 1] = 1
                #         print(x_min, x_max, y_min, y_max)
                masks.append(mask.flatten())
        masks = torch.stack(masks, dim=0).unsqueeze(dim=0)
        return masks

    def extract_patch_weights(self, heatmaps, patch_size, weight_min=0.1, weight_max=1.0,
                              heatmap_binary_threshold=None):
        # heatmap = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_AREA)
        # heatmap = heatmap[:, :, 0]  # only need one channel for mask
        # heatmap = heatmap.astype(np.float32)
        assert heatmaps.dim() == 4
        heatmaps = heatmaps[:, :, :, 0]

        if heatmap_binary_threshold is not None:
            if heatmap_binary_threshold == 'mean':
                # print('Using mean to binarize the heatmap weights')
                threshold = heatmaps.mean(dim=[1, 2], keepdim=True)
            else:
                raise NotImplementedError
            heatmaps = (heatmaps > threshold)
        N = heatmaps.shape[0]
        h = w = heatmaps.shape[1] // patch_size
        heatmaps = heatmaps.reshape(N, h, patch_size, w, patch_size)
        heatmaps = torch.einsum('nhpwq->nhwpq', heatmaps)
        heatmap_weights = heatmaps.reshape(N, h * w, patch_size ** 2).sum(dim=-1)
        heatmap_weights = (heatmap_weights / heatmap_weights.max(dim=1, keepdim=True)[0]) * (
                weight_max - weight_min) + weight_min
        return heatmap_weights

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, heatmaps=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.mask_strategy in ['heatmap_weighted', 'heatmap_inverse_weighted']:
            # print(self.mask_strategy)
            assert heatmaps is not None
            # print(heatmaps.shape)
            heatmap_weights = self.extract_patch_weights(heatmaps, self.patch_size, weight_min=self.weight_range[0],
                                                         weight_max=self.weight_range[1],
                                                         heatmap_binary_threshold=self.heatmap_binary_threshold)
            # print(heatmap_weights.shape)
            # assert self.heatmap_weights is not None

        if self.mask_strategy == 'heatmap_weighted':
            weight = heatmap_weights.to(x.device)  # length 1D vector
            noise = noise * weight
        elif self.mask_strategy == 'heatmap_inverse_weighted':
            weight = heatmap_weights.to(x.device)  # length 1D vector
            weight = (weight.max(dim=1, keepdim=True)[0] - weight) + weight.min(dim=1, keepdim=True)[0]
            noise = noise * weight
        # print(heatmaps[0][0])
        if 'local' in self.mask_strategy:
            # print(self.mask_strategy)
            heatmaps = heatmaps * self.local_attention_mask

        if 'self_attention_mean' in self.mask_strategy:
            assert heatmaps.shape == (N, L, L)
            if 'square' in self.mask_strategy and 'weight_square' not in self.mask_strategy:
                heatmaps = torch.square(heatmaps)
            elif 'sqrt' in self.mask_strategy and 'weight_sqrt' not in self.mask_strategy:
                heatmaps = torch.sqrt(heatmaps)

            if 'local' in self.mask_strategy and 'global' not in self.mask_strategy:
                heatmaps_local = heatmaps * self.local_attention_mask
                weight = heatmaps_local.sum(dim=-1) / self.local_attention_mask.sum(dim=-1)
            elif 'local' in self.mask_strategy and 'global' in self.mask_strategy:
                heatmaps_local = heatmaps * self.local_attention_mask
                weight_local = heatmaps_local.sum(dim=-1) / self.local_attention_mask.sum(dim=-1)
                weight_global = heatmaps.mean(dim=-1)
                weight = (weight_local + weight_global) / 2
            else:
                weight = heatmaps.mean(dim=-1)

            if 'weight_square' in self.mask_strategy:
                weight = torch.square(weight)
            elif 'weight_sqrt' in self.mask_strategy:
                weight = torch.sqrt(weight)

            if 'inverse' in self.mask_strategy:
                weight = (weight.max(dim=1, keepdim=True)[0] - weight) + weight.min(dim=1, keepdim=True)[0]

            noise = noise * weight

        if 'cross_batch_attention_mean' in self.mask_strategy:
            assert heatmaps.shape == (N, L, N * L)
            if 'local' in self.mask_strategy:
                # print(self.mask_strategy)
                weight = heatmaps.sum(dim=-1) / self.local_attention_mask.sum(dim=-1)
            else:
                weight = heatmaps.mean(dim=-1)

            if 'inverse' in self.mask_strategy:
                weight = (weight.max(dim=1, keepdim=True)[0] - weight) + weight.min(dim=1, keepdim=True)[0]
            noise = noise * weight

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, heatmaps=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        if 'self_attention' in self.mask_strategy and 'without_pe' in self.mask_strategy:
            # [N, L, H * W]
            feat = F.normalize(x, dim=-1)

            # [N, L, L]
            attention = torch.bmm(feat, feat.transpose(1, 2))
            attention = torch.clamp(attention, min=0.)
            heatmaps = attention

        x = x + self.pos_embed[:, 1:, :]

        if 'cross_batch_attention' in self.mask_strategy and 'without_pe' not in self.mask_strategy:
            # [N, L, c]
            feat = F.normalize(x, dim=-1)
            N = feat.shape[0]
            L = feat.shape[1]
            # [N, L, L]
            feat = feat.reshape(-1, feat.shape[-1])
            attention = torch.mm(feat, feat.transpose(0, 1))
            # attention = torch.bmm(feat, feat.transpose(1, 2))
            attention = torch.clamp(attention, min=0.)
            attention = attention.reshape(N, L, N * L)
            heatmaps = attention

        if 'self_attention' in self.mask_strategy and 'without_pe' not in self.mask_strategy:
            # [N, L, H * W]
            feat = F.normalize(x, dim=-1)

            # [N, L, L]
            attention = torch.bmm(feat, feat.transpose(1, 2))
            attention = torch.clamp(attention, min=0.)
            heatmaps = attention
        # masking: length -> length * mask_ratio

        x, mask, ids_restore = self.random_masking(x, mask_ratio, heatmaps)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, heatmaps=None):
        if heatmaps is not None:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, heatmaps)
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_small_patch16_dec128d2b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch16_dec512d2b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch16_dec512d2b_448(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=448, patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d2b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16_dec512d8 = mae_vit_small_patch16_dec512d8b
mae_vit_small_patch16_dec512d2 = mae_vit_small_patch16_dec512d2b
mae_vit_small_patch16 = mae_vit_small_patch16_dec128d2b
mae_vit_base_patch16_dec512d2 = mae_vit_base_patch16_dec512d2b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
