import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=heads, batch_first=True)

    def forward(self, query, key, value):
        # 输入：query, key, value = (B, D)
        attn_output, _ = self.attn(query, key, value)  # query, key, value: (B, L, D)

        return attn_output
