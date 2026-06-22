import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, feat_dim, deep_dim, hidden_dim=128):
        super().__init__()

        # project về cùng space
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.deep_proj = nn.Linear(deep_dim, hidden_dim)

        # attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, feat, deep):
        feat_p = self.feat_proj(feat)
        deep_p = self.deep_proj(deep)

        x = torch.cat([feat_p, deep_p], dim=1)

        weights = torch.softmax(self.attention(x), dim=1)

        w_feat = weights[:, 0].unsqueeze(1)
        w_deep = weights[:, 1].unsqueeze(1)

        fused = w_feat * feat_p + w_deep * deep_p

        return fused