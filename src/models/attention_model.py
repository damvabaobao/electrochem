import torch.nn as nn
from .attention_fusion import AttentionFusion

class AttentionModel(nn.Module):
    def __init__(self, feat_dim, deep_dim, num_classes):
        super().__init__()

        self.fusion = AttentionFusion(feat_dim, deep_dim)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),   # vì bạn đã fix fusion
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, feat, deep):
        fused = self.fusion(feat, deep)
        return self.classifier(fused)