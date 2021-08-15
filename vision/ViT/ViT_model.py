
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
                 fake_init: bool = False):
        super(PatchEmbedding, self).__init__()

        assert image_width % patch_size == image_height % patch_size == 0

        self.hidden_dim = hidden_dim

        self.linear_patch_projection = nn.Conv2d(in_channels=n_channels,
                                                 out_channels=self.hidden_dim,
                                                 kernel_size=patch_size,
                                                 stride=patch_size)
        # initialization use only for testing code
        if fake_init:
            self.linear_patch_projection.weight.data.fill_(1.)
            self.linear_patch_projection.bias.data.fill_(0.)

        self.pos_embed = nn.Parameter(torch.zeros(
            (image_height*image_width)//(patch_size**2), self.hidden_dim))

    def forward(self, x):
        x = self.linear_patch_projection(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x + self.pos_embed


class SelfAttention(nn.Module):

    def __init__(self, input_dimension: int, hiden_dimension: int):
        super(SelfAttention, self).__init__()

        self.W_q = nn.Linear(
            input_dimension, hiden_dimension, bias=False).cuda()
        self.W_k = nn.Linear(
            input_dimension, hiden_dimension, bias=False).cuda()
        self.W_v = nn.Linear(
            input_dimension, hiden_dimension, bias=False).cuda()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        return self.softmax(torch.matmul(Q, torch.transpose(K, 1, 2)) / ((Q.shape[1])**(1/2))) @ V


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dimension: int, mlp_dim: int, n_head: int):
        super(MultiHeadAttention, self).__init__()

        self.norm = nn.LayerNorm(input_dimension)

        self.self_attention_layers = []
        for _ in range(n_head):
            self.self_attention_layers.append(
                SelfAttention(input_dimension, input_dimension//n_head))

        self.mlp = nn.Sequential(nn.LayerNorm(input_dimension),
                                 nn.Linear(input_dimension, mlp_dim),
                                 nn.GELU(),
                                 nn.Linear(mlp_dim, input_dimension))

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


class VisionTransformer(nn.Module):

    def __init__(self, patch_size: int,
                 hidden_dim: int,
                 n_layers: int,
                 num_classes: int,
                 mlp_dim: int,
                 n_head: int,
                 image_size: Tuple[int, int, int]):
        super(VisionTransformer, self).__init__()

        n_channels, image_height, image_width = image_size
        self.patch_embeding = PatchEmbedding(patch_size,
                                             hidden_dim,
                                             image_height,
                                             image_width,
                                             n_channels)

        layers = []
        for _ in range(n_layers):
            layers.append(MultiHeadAttention(
                hidden_dim, mlp_dim, n_head))

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.patch_embeding(x)
        x = self.layers(x)

        # in original paper was presented special patch token [CLS]
        x = x.mean(-2)
        return self.fc(x)


def vit(patch_size: int = 8,
        hidden_dim: int = 256,
        n_layers: int = 6,
        num_classes: int = 10,
        mlp_dim: int = 512,
        n_head: int = 8,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        pretrained=False) -> VisionTransformer:

    model = VisionTransformer(patch_size,
                              hidden_dim,
                              n_layers,
                              num_classes,
                              mlp_dim,
                              n_head,
                              image_size)

    if pretrained:
        model.load_state_dict(torch.load("pretrained_vit.pth"))
        model.eval()
    return model
