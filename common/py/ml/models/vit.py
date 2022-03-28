import math

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def show_patches(args, img):
    tpl = "c (h ph) (w pw) -> h w c ph pw"
    patches = einops.rearrange(img,
                               tpl,
                               ph=args.patch_size,
                               pw=args.patch_size)
    nh, nw = patches.shape[0], patches.shape[1]

    ax = plt.subplot2grid((nh, nw * 2), (0, 0), rowspan=nh, colspan=nw)
    ax.imshow(transforms.ToPILImage()(img))
    for h in range(nh):
        for w in range(nw):
            ax = plt.subplot2grid((nw, nw * 2), (h, nw + w))
            ax.imshow(transforms.ToPILImage()(patches[h][w]))
    plt.show()


class PatchEmbedding(nn.Module):
    def __init__(self, n_channels, h, w, n_embedding, patch_size=16, **_):
        super().__init__()
        self.conv2d = nn.Conv2d(n_channels,
                                n_embedding,
                                kernel_size=patch_size,
                                stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, n_embedding))
        self.positions = nn.Parameter(
            torch.randn((h // patch_size) * (w // patch_size) + 1,
                        n_embedding))

    def forward(self, x):
        x = self.conv2d(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        cls_tokens = einops.repeat(self.cls_token,
                                   "n e -> b n e",
                                   b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class Attention(nn.Module):
    def __init__(self, n_embedding, n_heads=8, attention_dropout=0.0, **_):
        super().__init__()
        self.n_embedding = n_embedding
        self.n_heads = n_heads
        self.keys = nn.Linear(self.n_embedding, self.n_embedding)
        self.queries = nn.Linear(self.n_embedding, self.n_embedding)
        self.values = nn.Linear(self.n_embedding, self.n_embedding)
        self.dropout = nn.Dropout(attention_dropout)
        self.linear = nn.Linear(self.n_embedding, self.n_embedding)

    def forward(self, x, mask=None):
        tpl = "b n (h d) -> b h n d"
        queries = einops.rearrange(self.queries(x), tpl, h=self.n_heads)
        keys = einops.rearrange(self.keys(x), tpl, h=self.n_heads)
        values = einops.rearrange(self.values(x), tpl, h=self.n_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            energy.mask_fill(~mask, torch.finfo(torch.float32).min)

        attention = F.softmax(energy, dim=-1) / math.sqrt(self.n_embedding)
        attention = self.dropout(attention)
        x = torch.einsum("bhqk, bhkd -> bhqd", attention, values)
        x = einops.rearrange(x, "b h n d -> b n (h d)")
        x = self.linear(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x, **kwargs):
        return x + self.f(x, **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, n_embedding, expansion=4, forward_dropout=0.0, **_):
        super().__init__(
            nn.Linear(n_embedding, expansion * n_embedding),
            nn.GELU(),
            nn.Dropout(forward_dropout),
            nn.Linear(expansion * n_embedding, n_embedding),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, n_embedding, dropout=0.0, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(n_embedding),
                              Attention(n_embedding, **kwargs),
                              nn.Dropout(dropout))),
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(n_embedding),
                              FeedForward(n_embedding, **kwargs),
                              nn.Dropout(dropout))))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, n_embedding, n_classes):
        super().__init__(nn.LayerNorm(n_embedding),
                         nn.Linear(n_embedding, n_classes))

    def forward(self, x):
        return super().forward(x.mean(1))


class ViT(nn.Sequential):
    def __init__(self, n_channels, h, w, n_embedding, depth, n_classes,
                 **kwargs):
        super().__init__(
            PatchEmbedding(n_channels, h, w, n_embedding, **kwargs),
            TransformerEncoder(depth, n_embedding=n_embedding, **kwargs),
            ClassificationHead(n_embedding, n_classes))
