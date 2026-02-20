import torch
import torch.nn as nn
import torch.nn.functional as F

class RawNetLite(nn.Module):
    def __init__(self, distill_dim=768):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)

        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)

        self.pool = nn.AdaptiveAvgPool1d(64)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.embed_fc = nn.Linear(256, distill_dim)
        self.cls_fc = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))

        x = self.pool(x)
        x = x.transpose(1, 2)

        out, _ = self.gru(x)
        feat = out[:, -1, :]   # [B, 256]

        embed = self.embed_fc(feat)
        logits = self.cls_fc(feat)

        return logits, embed
