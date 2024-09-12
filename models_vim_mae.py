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


from timm.models.vision_transformer import PatchEmbed, DropPath, _cfg

from utils.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed, FactorizedPositionalEncoding
from models_vim import PatchEmbed3D, create_block, _init_weights, PatchEmbedUniversial

from rotary_embedding_torch import RotaryEmbedding
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MaskedAutoencoderVim(nn.Module):
    """ Masked Autoencoder with VisionMamba backbone
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24, 
               
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 ssm_cfg=None, 
                 
                encoder_attn_layer_idx=None,
                decoder_attn_layer_idx=None,
                attn_cfg=None,
                 
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
     
                 pt_hw_seq_len=14,
                 if_rope=False,
                 if_rope_residual=False,
                 bimamba_type="none",
                 mixer_type="mamba",
                 if_devide_out = True, 
                norm_pix_loss=False,
                if_3d=False,
                if_cls_token = False,
            
                
                 **kwargs
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
       
        self.num_tokens = 1 if if_cls_token else 0
        
        self.if_3d = if_3d
        self.if_cls_token = if_cls_token

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, 
                                      norm_layer= nn.LayerNorm if not self.if_cls_token else None) if not if_3d else PatchEmbed3D(
            volume_size=img_size,
            patch_size= patch_size,
            in_chans= in_chans,
            embed_dim= embed_dim, 
            norm_layer= nn.LayerNorm if not self.if_cls_token else None)
        self.in_chans = in_chans
        # self.patch_embed = PatchEmbedUniversial(
        #     img_size = img_size,
        #     in_chans = in_chans,
        #     embed_dim = embed_dim,
        #     norm_layer = nn.LayerNorm if not self.if_cls_token else None,
        #     if_3d = if_3d
        # )
        self.d_model = self.num_features = self.embed_dim = embed_dim
        num_patches = self.patch_embed.num_patches
        
        # remove cls token like in Vmamba
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if if_rope:
            half_head_dim = embed_dim // 2
            self.rope = RotaryEmbedding(dim = half_head_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
                # transformer blocks
        self.blocks = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=encoder_attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out = if_devide_out,
                    mixer_type= mixer_type,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        decoder_inter_dpr = [0.0] + decoder_dpr
        self.decoder_drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.decoder_blocks = nn.ModuleList(
            [
                create_block(
                    decoder_embed_dim,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=decoder_attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=decoder_inter_dpr[i],
                    if_devide_out = if_devide_out,
                    mixer_type= mixer_type,
                    **factory_kwargs,
                )
                for i in range(decoder_depth)
            ]
        )
        if if_rope:
            half_head_dim = decoder_embed_dim // 2
            
            self.decoder_rope = RotaryEmbedding(dim = half_head_dim)
        
        self.decoder_norm_f =  (nn.LayerNorm if not rms_norm else RMSNorm)(
            decoder_embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        
        self.decoder_pred = nn.Linear(decoder_embed_dim, 
                                      patch_size**2 * in_chans,
                                      bias=True) if not self.if_3d else nn.Linear(decoder_embed_dim,
                                                                                  patch_size**3 * in_chans,
                                                                                  bias=True)# decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        # original init 
        self.initialize_weights()
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.if_3d:
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], round(self.patch_embed.num_patches ** (1 / 3)),
                                            # int(self.patch_embed.num_patches ** (1/3)),
                                            cls_token= self.if_cls_token) 
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
            #                                             int(self.patch_embed.num_patches ** .5), cls_token=True)
            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                        round(self.patch_embed.num_patches ** (1 / 3)), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=self.if_cls_token)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
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
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], -1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, -1))
        return x

    def patchify_3d(self, volume):
        """
        volume: (N, 3, L, H, W)
        x: (N, L, patch_size**3 *3)
        """
        p = self.patch_embed.patch_size[0]  # Patch size
        assert volume.shape[2] == volume.shape[3] == volume.shape[4] and volume.shape[
            2] % p == 0  # Ensuring we have the same dimension

        l = h = w = volume.shape[2] // p  # Since volumes have the same dimension. Possible limitation??
        x = volume.reshape(shape=(volume.shape[0], -1, l, p, h, p, w, p))
        x = torch.einsum('nclrhpwq->nlhwrpqc', x)
        x = x.reshape(shape=(volume.shape[0], l * h * w, -1))
        return x

    
    def unpatchify_3d(self, x):
        """
        x: (N, L, patch_size**3 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        l = h = w = round(x.shape[1] ** (1 / 3))
        assert l * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], l, h, w, p, p, p,
                             -1))  # Earlier 3 was hard-coded here. Maybe this way, we are more flexible with the number of channels
        x = torch.einsum('nlhwrpqc->nclrhpwq', x)
        volume = x.reshape(shape=(x.shape[0], -1, h * p, h * p, h * p))
        return volume

    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
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

    def forward_encoder(self, x, mask_ratio, inference_params=None):
        # embed patches
        # print("x1 shape", x.shape)
        x = self.patch_embed(x)
        # print("patch embedding shape", x.shape)

        # add pos embed w/o cls token
        if not self.if_cls_token:
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed[:, 1:, :]
        # print("x + position embedding", x.shape)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # print("after random mask x shape", x.shape)
        # append cls token
        if self.if_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            # print("cls shape", cls_tokens.shape)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply VIM blocks
        residual = None
        # print("encoder input x shape", x.shape)
        hidden_states = x
        for layer in self.blocks:
            # rope about
            if self.if_rope:
                hidden_states = self.rope.rotate_queries_or_keys(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope.rotate_queries_or_keys(residual)
            # print('encoder hidden state shape', hidden_states.shape)
            # # print('encoder norm shape', self.norm_f.weight.shape)
            # print("residual shape", residual.shape if residual is not None else None)
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
             hidden_states = layer_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm_f.eps,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        
        return hidden_states, mask, ids_restore

    def forward_decoder(self, x, ids_restore, inference_params=None):
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
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        # x = self.decoder_norm(x)
        residual = None
        hidden_states = x
        for layer in self.decoder_blocks:
            # rope about
            if self.if_rope:
                hidden_states = self.decoder_rope.rotate_queries_or_keys(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.decoder_rope.rotate_queries_or_keys(residual)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.decoder_drop_path(hidden_states)
            hidden_states = self.decoder_norm_f(residual.to(dtype=self.decoder_norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.decoder_norm_f, RMSNorm) else layer_norm_fn
            # print(self.decoder_norm_f.type)
            # print('decoder hidden state shape', hidden_states.shape)
            # print('decoder_norm_f weight shape',self.decoder_norm_f.weight.shape)
            hidden_states = fused_add_norm_fn(
                self.decoder_drop_path(hidden_states),
                self.decoder_norm_f.weight,
                self.decoder_norm_f.bias,
                eps=self.decoder_norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # predictor projection
        pred = self.decoder_pred(hidden_states)

        # remove cls token
        pred = pred[:, 1:, :]

        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.if_3d:
            target = self.patchify_3d(imgs)
        else:
            target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
     
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vim_small_patch16_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, 
        embed_dim=384, depth=12,
        decoder_embed_dim=192, decoder_depth=8,
        rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2", 
        if_devide_out = True, 
        if_cls_token= True,
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_vim_small_patch4_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=4, 
        embed_dim=384, depth=12,
        decoder_embed_dim=192, decoder_depth=8,
        rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2", 
        if_devide_out = True, 
        if_cls_token= True,
        **kwargs
        )
    model.default_cfg = _cfg()
    return model


def mae_vim_small_patch8_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=8, 
        embed_dim=384, depth=12,
        decoder_embed_dim=192, decoder_depth=8,
        rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2", 
        if_devide_out = True, 
        if_cls_token= True,
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_vim_base_patch16_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, embed_dim=768, depth=24,
        decoder_embed_dim=192, decoder_depth=12,
        rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual= True,
        bimamba_type="v2", 
        if_devide_out = True,
        if_cls_token= True,
        **kwargs
        )
    model.default_cfg = _cfg()
    return model


def mae_vim_large_patch16_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, embed_dim=1024, depth=24,
        decoder_embed_dim=192, decoder_depth=12,
        rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim_small_patch16_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, embed_dim=384, depth=12,
        decoder_embed_dim=192, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim_small_patch4_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=4, embed_dim=384, depth=12,
        decoder_embed_dim=192, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim_small_patch8_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=8, embed_dim=384, depth=12,
        decoder_embed_dim=192, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim_small_patch32_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=32, embed_dim=384, depth=12,
        decoder_embed_dim=192, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim_base_patch16_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, embed_dim=768, depth=12,
        decoder_embed_dim=512, decoder_depth=8,
        rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        **kwargs
        )
    model.default_cfg = _cfg()
    return model



def mae_3d_vim_large_patch16_rms_bimambav2_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, embed_dim=1024, depth=24,
        decoder_embed_dim=512, decoder_depth=8,
        rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=True,
        if_rope_residual=True,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim2_small_patch32_rms_no_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=32, embed_dim=512, depth=12,
        decoder_embed_dim=256, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        mixer_type="mamba2",
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim2_small_patch16_rms_no_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, embed_dim=512, depth=12,
        decoder_embed_dim=256, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        mixer_type="mamba2",
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_vim2_small_patch4_rms_no_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=4, embed_dim=512, depth=12,
        decoder_embed_dim=256, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        mixer_type="mamba2",
        **kwargs
        )
    model.default_cfg = _cfg()
    return model

def mae_3d_mha_small_patch16_rms_no_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=16, embed_dim=512, depth=12,
        decoder_embed_dim=256, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        mixer_type="mamba2",
        encoder_attn_layer_idx=[3,6,9],
        decoder_attn_layer_idx=[],
        attn_cfg={
        "causal": False,
        "d_conv": 4,
        "head_dim": 128,
        "num_heads": 16,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": 0,
        },
        **kwargs
        )
    model.default_cfg = _cfg()
    return model



def mae_3d_mha_small_patch4_rms_no_rope_residual(**kwargs):
    model = MaskedAutoencoderVim(
        patch_size=4, embed_dim=512, depth=12,
        decoder_embed_dim=256, decoder_depth=8,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_devide_out = True,
        if_cls_token= False, 
        if_3d = True, 
        mixer_type="mamba2",
        encoder_attn_layer_idx=[3,6,9],
        decoder_attn_layer_idx=[],
        attn_cfg={
        "causal": False,
        "d_conv": 4,
        "head_dim": 128,
        "num_heads": 16,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": 0,
        },
        **kwargs
        )
    model.default_cfg = _cfg()
    return model



# set recommended archs
mae_vim_small_patch4 = mae_vim_small_patch4_rms_bimambav2_rope_residual  # decoder: 384 dim, 12 blocks
mae_vim_small_patch8 = mae_vim_small_patch8_rms_bimambav2_rope_residual
mae_vim_small_patch16 = mae_vim_small_patch16_rms_bimambav2_rope_residual  # decoder: 384 dim, 12 blocks
mae_vim_base_patch16 = mae_vim_base_patch16_rms_bimambav2_rope_residual  # decoder: 768 dim, 12 blocks
mae_vim_large_patch14 = mae_vim_large_patch16_rms_bimambav2_rope_residual  # decoder: 1024 dim, 24 blocks

# set 3d models
mae_3d_vim_small_patch8 = mae_3d_vim_small_patch8_rms_bimambav2_rope_residual
mae_3d_vim_small_patch16 = mae_3d_vim_small_patch16_rms_bimambav2_rope_residual
mae_3d_vim_base_patch16 = mae_3d_vim_base_patch16_rms_bimambav2_rope_residual
mae_3d_vim_large_patch16 = mae_3d_vim_large_patch16_rms_bimambav2_rope_residual
mae_3d_vim_small_patch4 = mae_3d_vim_small_patch4_rms_bimambav2_rope_residual
mae_3d_vim_small_patch32 = mae_3d_vim_small_patch32_rms_bimambav2_rope_residual

# set 3d mamba2 models
mae_3d_vim2_small_patch32 = mae_3d_vim2_small_patch32_rms_no_rope_residual
mae_3d_vim2_small_patch16 = mae_3d_vim2_small_patch16_rms_no_rope_residual
mae_3d_vim2_small_patch4 = mae_3d_vim2_small_patch4_rms_no_rope_residual

# set 3d mha models
mae_3d_vim_mha_small_patch16 = mae_3d_mha_small_patch16_rms_no_rope_residual

mae_3d_vim_mha_small_patch4 = mae_3d_mha_small_patch4_rms_no_rope_residual

