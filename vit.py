import torch
import torch.nn as nn
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_height, patch_width, emb_size) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_height * patch_width * in_channels),
            nn.Linear(patch_height * patch_width * in_channels, emb_size),
            nn.LayerNorm(emb_size),
        )
    def forward(self, x):
        """
        B x C x H x W -> B x N x emb
        N = Number of patches
        """
        return self.projection(x)

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output


class FeedForward(nn.Sequential):
    def __init__(self, dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, dropout) -> None:
        super().__init__()

        self.att = nn.Sequential(nn.LayerNorm(dim), Attention(dim, n_heads, dropout))
        self.ff = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, dropout))
    def forward(self, x):
        x = x + self.att(x)
        x = x + self.ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, ch, img_size, patch_size, emb_dim,
                n_layers, out_dim, dropout=0.1, heads=2):
        super(ViT, self).__init__()
        self.channels = ch
        self.n_patches = (img_size // patch_size) ** 2
        self.emb_dim = emb_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(in_channels=self.channels,
                                          patch_height=patch_size, patch_width=patch_size,
                                          emb_size=self.emb_dim)
        
        self.positional_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, self.emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim)) # shape of a single embedded patch

        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(TransformerEncoder(dim=self.emb_dim,
                                                    n_heads=heads,
                                                    dropout=dropout
                                                    ))
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

    def forward(self, x):
        x = self.patch_embed(x)
        
        b, _, _ = x.shape # B x N x EMB
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_token, x], dim=1)

        x += self.positional_embedding
        
        for encoder in self.encoders:
            x = encoder(x)
        
        return self.head(x[:, 0, :]) # pass the cls_token to classifier
