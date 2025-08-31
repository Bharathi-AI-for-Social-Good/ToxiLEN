import torch
import torch.nn as nn

class BiGateFusion(nn.Module):
    """
    输入:  feat_a, feat_b  (B, D)  —— 两路同维度特征
    输出: fused           (B, 2D) —— 拼接双门控结果
    """
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, hidden_dim)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))

        self.last_score1 = None
        self.last_score2 = None

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        # gate scores
        # feat_a = clip_img and clip_text
        # feat_b = span_feat

        score1 = torch.sigmoid(self.W1(feat_a) + torch.tanh(self.W2(feat_b)) + self.b1)  # span -> img/text
        score2 = torch.sigmoid(self.W3(feat_b) + torch.tanh(self.W4(feat_a)) + self.b2)  # img/text -> span

        self.last_score1 = score1
        self.last_score2 = score2

        VS = score1 * feat_a
        SV = score2 * feat_b
        fused = torch.cat([VS, SV], dim=-1)                                              # (B,2D)
        return fused
