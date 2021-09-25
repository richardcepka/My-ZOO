""" This code is only for educational purposes.

---
ViT implementation https://arxiv.org/pdf/2010.11929.pdf with optional mean clasification head (default [CLS] token).

Code inspiration:
https://github.com/eemlcommunity/PracticalSessions2021/blob/main/vision/vision_transformers.ipynb
https://github.com/The-AI-Summer/self-attention-cv/blob/main/self_attention_cv/vit/vit.py
---

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


class SelfAttention(nn.Module):

    def __init__(self, input_dimension: int, hiden_dimension: int):
        super(SelfAttention, self).__init__()

        self.W_q = nn.Linear(
            input_dimension, hiden_dimension, bias=False)
        self.W_k = nn.Linear(
            input_dimension, hiden_dimension, bias=False)
        self.W_v = nn.Linear(
            input_dimension, hiden_dimension, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        return self.softmax(torch.matmul(Q, torch.transpose(K, 1, 2)) / ((Q.shape[1])**(1/2))) @ V


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dimension: int, mlp_dim: int, n_head: int, dropout: int):
        super(MultiHeadAttention, self).__init__()

        self.norm = nn.LayerNorm(input_dimension)

        self.self_attention_layers = nn.ModuleList(
            [SelfAttention(input_dimension, input_dimension//n_head) for _ in range(n_head)])

        self.mlp = nn.Sequential(nn.LayerNorm(input_dimension),
                                 nn.Linear(input_dimension, mlp_dim),
                                 nn.Dropout(dropout),
                                 nn.GELU(),
                                 nn.Linear(mlp_dim, input_dimension),
                                 nn.Dropout(dropout))

    def forward(self, x):
        identity = x

        x = self.norm(x)
        multi_head_attention_outputs = []
        for self_attention_layer in self.self_attention_layers:
            multi_head_attention_outputs.append(self_attention_layer(x))

        x = torch.cat(multi_head_attention_outputs, 2)
        x += identity

        y = self.mlp(x)

        return x + y


class ViT(nn.Module):

    def __init__(self, patch_size: int,
                 hidden_dim: int,
                 n_layers: int,
                 num_classes: int,
                 mlp_dim: int,
                 dropout: int,
                 n_head: int,
                 image_size: Tuple[int, int, int],
                 clasification_head: str = "cls"):
        super(ViT, self).__init__()

        assert clasification_head in ["cls", "mean"]
        self.clasification_head = clasification_head

        n_channels, image_height, image_width = image_size
        self.patch_embeding = PatchEmbedding(patch_size,
                                             hidden_dim,
                                             image_height,
                                             image_width,
                                             n_channels,
                                             clasification_head=clasification_head)

        layers = []
        for _ in range(n_layers):
            layers.append(MultiHeadAttention(
                hidden_dim, mlp_dim, n_head, dropout))

        self.layers = nn.Sequential(*layers)
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
