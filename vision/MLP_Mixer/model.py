""" This code is only for educational purposes.
"""

import torch.nn as nn
import torch
from einops import rearrange
from typing import Tuple


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int,
                 hidden_dim: int,
                 image_height: int,
                 image_width: int,
                 n_channels: int,
                 clasification_head: str = "cls"):
        super(PatchEmbedding, self).__init__()
        """
        If clasification_head == True, than use mean head clasification.
        """
        assert image_width % patch_size == image_height % patch_size == 0
        assert clasification_head in ["cls", "mean"]

        self.hidden_dim = hidden_dim
        self.clasification_head = clasification_head

        self.linear_patch_projection = nn.Conv2d(in_channels=n_channels,
                                                 out_channels=self.hidden_dim,
                                                 kernel_size=patch_size,
                                                 stride=patch_size)

        if self.clasification_head == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        tokens = (image_height*image_width)//(patch_size**2)
        tokens += 1 if self.clasification_head == "cls" else 0
        self.pos_embed = nn.Parameter(torch.zeros(tokens, self.hidden_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear_patch_projection(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        if self.clasification_head == "cls":
            cls_token_shape = self.cls_token.shape
            cls_token_shape = [batch_size] + list(cls_token_shape[1:])
            cls = self.cls_token.expand(cls_token_shape)
            x = torch.cat((cls, x), dim=1)

        return x + self.pos_embed


class MLP_Mixer_block(nn.Module):

    def __init__(self, input_dimension: int, n_tokens: int, output_dimension: int, mlp_dim: int, dropout: int):
        super(MLP_Mixer_block, self).__init__()

        self.mlp_1 = nn.Sequential(nn.Linear(n_tokens, mlp_dim),
                                   nn.GELU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(mlp_dim, n_tokens),
                                   nn.Dropout(dropout))

        self.mlp_2 = nn.Sequential(nn.Linear(input_dimension, mlp_dim),
                                   nn.Dropout(dropout),
                                   nn.GELU(),
                                   nn.Linear(mlp_dim, output_dimension),
                                   nn.Dropout(dropout))

        self.norm_1 = nn.LayerNorm(input_dimension)
        self.norm_2 = nn.LayerNorm(input_dimension)

    def forward(self, x):
        identity = x  # first skip-conection
        x = self.norm_1(x)  # first layer-norm
        x = rearrange(x, 'b h w -> b w h')  # token-mixing
        x = self.mlp_1(x)
        x = rearrange(x, 'b w h -> b h w')  # channel-mixing
        x = x + identity
        identity = x  # second skip-conection
        x = self.norm_2(x)  # second layer-norm
        x = self.mlp_2(x)
        return x + identity


class MLP_Mixer(nn.Module):

    def __init__(self, patch_size: int,
                 hidden_dim: int,
                 n_layers: int,
                 num_classes: int,
                 mlp_dim: int,
                 dropout: int,
                 image_size: Tuple[int, int, int],
                 clasification_head: str = "cls"):
        super(MLP_Mixer, self).__init__()

        assert clasification_head in ["cls", "mean"]
        self.clasification_head = clasification_head

        n_channels, image_height, image_width = image_size
        self.patch_embeding = PatchEmbedding(patch_size,
                                             hidden_dim,
                                             image_height,
                                             image_width,
                                             n_channels,
                                             clasification_head=clasification_head)

        n_tokens = (image_height*image_width)//(patch_size**2)
        n_tokens += 1 if self.clasification_head == "cls" else 0
        self.layers = nn.Sequential(*[MLP_Mixer_block(hidden_dim, n_tokens, hidden_dim, mlp_dim, dropout)
                                    for _ in range(n_layers)])

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embeding(x)
        x = self.layers(x)

        # [CLS] clasification head or mean clasification head
        if self.clasification_head == "cls":
            x = x[:, 0, :]
        elif self.clasification_head == "mean":
            x = x.mean(-2)

        return self.fc(x)
