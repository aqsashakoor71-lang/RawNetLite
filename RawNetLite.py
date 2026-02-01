import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Residual block (BASELINE)
# ------------------------------
class ResBlock(nn.Module):
    """
    Baseline 1D residual block:
    Conv1d -> BN -> ReLU -> Conv1d -> BN -> +residual -> ReLU
    Input/Output shape: [B, C, T]
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


# ------------------------------
# RawNetLite (BASELINE)
# ------------------------------
class RawNetLite(nn.Module):
    """
    RawNetLite baseline:
      - raw waveform input [B,1,T]
      - Conv1d + BN + ReLU
      - 3 x ResBlock(64)
      - AdaptiveAvgPool1d(64)
      - BiGRU (seq=64, feat=64) -> output last step
      - FC -> logits [B,1]

    IMPORTANT:
      - Returns LOGITS (not sigmoid)
      - Use BCEWithLogitsLoss in training
      - For probabilities: torch.sigmoid(logits)
    """
    def __init__(self):
        super(RawNetLite, self).__init__()
        self.conv_pre = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn_pre   = nn.BatchNorm1d(64)
        self.relu     = nn.ReLU()

        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64)
        self.resblock3 = ResBlock(64)

        # compress time dimension to fixed length for GRU
        self.pool = nn.AdaptiveAvgPool1d(64)  # output: [B, 64, 64]

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.bn_pre(self.conv_pre(x)))  # [B, 64, T]

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = self.pool(x)                              # [B, 64, 64]
        x = x.transpose(1, 2)                         # [B, 64, 64] -> [B, seq=64, feat=64]

        out, _ = self.gru(x)                          # [B, 64, 256]
        x = out[:, -1, :]                             # last time-step: [B, 256]

        x = self.fc1(x)                               # [B, 64]
        logits = self.fc2(x)                          # [B, 1]

        return logits                                  # ✅ LOGITS (NO sigmoid)
