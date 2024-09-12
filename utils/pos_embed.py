# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


class FactorizedPositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        """
        Initialize the factorized positional encoding module.
        
        Args:
        - d_model: Dimensionality of the model (and hence the positional encoding)
        - height: Number of patches in the height dimension
        - width: Number of patches in the width dimension
        """
        super(FactorizedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width

        # Create positional encodings for rows and columns
        self.row_pos_embed = nn.Parameter(torch.zeros(1, height, d_model // 2))
        self.col_pos_embed = nn.Parameter(torch.zeros(1, width, d_model // 2))

        # Initialize the positional encodings
        self._initialize_positional_encodings()

    def _initialize_positional_encodings(self):
        """
        Initialize positional encoding parameters.
        """
        nn.init.uniform_(self.row_pos_embed, -0.1, 0.1)
        nn.init.uniform_(self.col_pos_embed, -0.1, 0.1)

    def forward(self, x):
        """
        Forward pass for adding factorized positional encodings to input tensor.
        
        Args:
        - x: Input tensor of shape (batch_size, num_patches, d_model)
        
        Returns:
        - Output tensor with positional encodings added, same shape as input.
        """
        batch_size, num_patches, _ = x.shape
        
        # Assume square root of number of patches is an integer for simplicity
        assert int(num_patches ** 0.5) ** 2 == num_patches, "Number of patches must be a perfect square"

        # Reshape input to (batch_size, height, width, d_model)
        x = x.view(batch_size, self.height, self.width, self.d_model)
        
        # Split the d_model dimension for row and column positional embeddings
        row_embeddings = self.row_pos_embed.repeat(batch_size, 1, 1, 1)
        col_embeddings = self.col_pos_embed.repeat(batch_size, 1, 1, 1).transpose(1, 2)

        # Concatenate row and column embeddings along the d_model dimension
        pos_embeddings = torch.cat((row_embeddings, col_embeddings), dim=-1)

        # Add positional embeddings to the input tensor
        x = x + pos_embeddings

        # Flatten the input back to (batch_size, num_patches, d_model)
        x = x.view(batch_size, num_patches, self.d_model)

        return x


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_l, grid_h, grid_w)  # Different from the github impl. Look at https://github.com/facebookresearch/mae/issues/18
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([-1, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use one third of dimensions to encode grid_l, grid_h and final dim for grid_w
    res = embed_dim // 3
    if res % 2 != 0:
        res += 1
    factor_w = embed_dim - 2 * res
    emb_l = get_1d_sincos_pos_embed_from_grid(res, grid[0])  # (L*H*W, D//3)
    emb_h = get_1d_sincos_pos_embed_from_grid(res, grid[1])  # (L*H*W, D//3)
    emb_w = get_1d_sincos_pos_embed_from_grid(factor_w, grid[2])  # (L*H*W, D-D//3-D//3)

    emb = np.concatenate([emb_l, emb_h, emb_w], axis=1)  # (L*H*W, D)
    return emb



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
