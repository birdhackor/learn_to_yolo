"""CPU 課堂用完整小 ViT：patch、位置、CLS、Pre-LN attention/MLP 與分類頭。

結構來源：Dosovitskiy et al., An Image is Worth 16x16 Words,
https://arxiv.org/abs/2010.11929 。這裡縮小圖、寬度及深度，並非原論文規模。
"""

import math

import torch
from torch import nn


def patchify(images, patch_size):
    """[B,3,H,W] -> [B,N,3*P*P]；patch 左至右、上至下，內部 C/row/col。"""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images 必須是 [B,3,H,W] RGB tensor")
    if (patch_size < 1 or images.shape[2] < 1 or images.shape[3] < 1
            or images.shape[2] % patch_size or images.shape[3] % patch_size):
        raise ValueError("H/W 必須能被正整數 patch_size 整除")
    batch, channels, height, width = images.shape
    return (images.reshape(batch, channels, height // patch_size, patch_size,
                           width // patch_size, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, (height // patch_size) * (width // patch_size),
                     channels * patch_size * patch_size))


class PatchEmbedding(nn.Module):
    """每個不重疊 P×P RGB patch 共用同一個線性投影，輸出 [B,N,D]。"""

    def __init__(self, image_size=32, patch_size=8, embed_dim=32):
        super().__init__()
        if image_size < 1 or patch_size < 1 or image_size % patch_size or embed_dim < 1:
            raise ValueError("image_size/patch_size/embed_dim 必須為正且圖尺寸可整除")
        self.image_size, self.patch_size, self.embed_dim = image_size, patch_size, embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"images 必須是 [B,3,{self.image_size},{self.image_size}]")
        return self.projection(images).flatten(2).transpose(1, 2)


class SelfAttention(nn.Module):
    """多頭 QK-softmax-V；權重 [B,heads,query,key]，每個 query 的 key 權重和為1。

    return_attention=True 回傳 dropout 前的權重，方便核對 softmax；訓練輸出會使用 dropout。
    """

    def __init__(self, embed_dim=32, num_heads=4, dropout=0.0):
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim 必須能被正整數 num_heads 整除")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必須在 [0,1)")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens, return_attention=False):
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim or tokens.shape[1] < 1:
            raise ValueError("tokens 必須是非空的 [B,N,embed_dim]")
        batch, count, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)
        mixed = (self.attention_dropout(weights) @ v).transpose(1, 2).reshape(batch, count, self.embed_dim)
        output = self.output_dropout(self.projection(mixed))
        return (output, weights) if return_attention else output


class TransformerBlock(nn.Module):
    """先 LayerNorm 再 attention/MLP，兩條 residual 保留原 token 的訊息。"""

    def __init__(self, embed_dim=32, num_heads=4, mlp_ratio=2, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        if hidden < 1:
            raise ValueError("MLP hidden dimension 必須為正")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, embed_dim), nn.Dropout(dropout))

    def forward(self, tokens):
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class TinyViT(nn.Module):
    """固定 RGB 圖尺寸的小型 ViT；features 同時保留 CLS 與逐 patch 表示。

    CLS 是序列第0個可學習向量。後面 N 個 patch 按列排列；learned pos_embed
    為每個位置各存一個向量。forward_features 並不 detach，供分類或自監督訓練。
    """

    def __init__(self, image_size=32, patch_size=8, embed_dim=32, depth=2,
                 num_heads=4, mlp_ratio=2, num_classes=2, dropout=0.0):
        super().__init__()
        if depth < 1 or num_classes < 1:
            raise ValueError("depth/num_classes 必須為正")
        self.image_size, self.patch_size, self.embed_dim = image_size, patch_size, embed_dim
        self.depth, self.num_heads, self.num_classes = depth, num_heads, num_classes
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def patchify(self, images):
        return patchify(images, self.patch_size)

    def embed_tokens(self, images, use_position=True):
        patches = self.patch_embed(images)
        tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), patches), dim=1)
        if use_position:
            tokens = tokens + self.pos_embed
        return self.embedding_dropout(tokens)

    def encode_tokens(self, tokens):
        """供教學介入：輸入已含 CLS/位置的 [B,1+N,D]，完成所有 blocks 和最終LN。"""
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (1 + self.num_patches, self.embed_dim):
            raise ValueError("tokens shape 必須與這個模型的 CLS+patch 數及 embed_dim 一致")
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_features(self, images, use_position=True):
        tokens = self.encode_tokens(self.embed_tokens(images, use_position=use_position))
        return {"cls": tokens[:, 0], "patches": tokens[:, 1:], "tokens": tokens}

    def forward(self, images):
        return self.head(self.forward_features(images)["cls"])
