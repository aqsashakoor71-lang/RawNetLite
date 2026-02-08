# rawnetlite.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Pre-Emphasis Layer
# ---------------------------
class PreEmphasis(nn.Module):
    """
    y[t] = x[t] - a * x[t-1]
    Helps reduce channel/codec bias
    """
    def __init__(self, a=0.97):
        super().__init__()
        self.register_buffer("a", torch.tensor(a, dtype=torch.float32))

    def forward(self, x):
        # x: [B, 1, T]
        x_prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
        return x - self.a * x_prev


# ---------------------------
# Residual Block
# ---------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


# ---------------------------
# Attentive Pooling
# ---------------------------
class AttentivePool(nn.Module):
    """
    Learns which time frames matter
    Input:  [B, T, C]
    Output: [B, C]
    """
    def __init__(self, channels, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        e = self.fc2(torch.tanh(self.fc1(x)))   # [B, T, 1]
        a = torch.softmax(e, dim=1)             # [B, T, 1]
        return torch.sum(a * x, dim=1)          # [B, C]


# ---------------------------
# RawNetLite (A3 variant)
# ---------------------------
class RawNetLite(nn.Module):
    """
    Input : [B, 1, T]
    Output: logits [B, 1]  (NO sigmoid here)
    """
    def __init__(self):
        super().__init__()

        self.preemph = PreEmphasis(0.97)

        self.conv_pre = nn.Conv1d(1, 64, 3, padding=1)
        self.bn_pre   = nn.BatchNorm1d(64)
        self.relu     = nn.ReLU(inplace=True)

        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)

        self.pool = nn.AdaptiveAvgPool1d(64)   # -> [B, 64, 64]

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attn = AttentivePool(256)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 1, T]
        x = self.preemph(x)

        x = self.relu(self.bn_pre(self.conv_pre(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.pool(x)           # [B,64,64]
        x = x.transpose(1, 2)      # [B,64,64] -> [B,T=64,F=64]

        out, _ = self.gru(x)       # [B,64,256]
        x = self.attn(out)         # [B,256]

        x = self.fc1(x)
        x = self.fc2(x)            # logits
        return x
