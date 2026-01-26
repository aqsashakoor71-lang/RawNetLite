import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual block base
class ResBlock(nn.Module):
    """
    A 1D convolutional residual block for processing sequential data.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


# RawNetLite model (MULTI-HEAD: CM + QUALITY)
class RawNetLite(nn.Module):
    """
    RawNetLite with multi-head outputs:
      - CM head: spoof vs bonafide (binary)
      - Quality head: clean vs degraded (2-class)
    """
    def __init__(self):
        super(RawNetLite, self).__init__()
        self.conv_pre = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn_pre = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64)
        self.resblock3 = ResBlock(64)

        self.pool = nn.AdaptiveAvgPool1d(64)  # Sequence reduction for GRU

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(128 * 2, 64)

        # ✅ CM head (binary spoof logit)
        self.fc2 = nn.Linear(64, 1)

        # ✅ Quality head (clean/degraded logits)
        self.fc_q = nn.Linear(64, 2)

    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.bn_pre(self.conv_pre(x)))     # [B, 64, T]
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.pool(x)                                 # [B, 64, 64]

        x = x.transpose(1, 2)                            # [B, 64, 64] → [B, seq, feat]
        output, _ = self.gru(x)                          # [B, 64, 256]
        x = output[:, -1, :]                             # Last step → [B, 256]

        x = self.fc1(x)                                  # [B, 64]

        # -------------------------------
        # ❌ OLD (single-head)
        # x = self.fc2(x)                                  # [B, 1]
        # return torch.sigmoid(x)
        # -------------------------------

        # ✅ NEW (multi-head)
        cm_logits = self.fc2(x)                           # [B, 1]
        q_logits  = self.fc_q(x)                          # [B, 2]
        return cm_logits, q_logits
