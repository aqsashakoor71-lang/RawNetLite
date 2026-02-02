# rawnetlite.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class RawNetLite(nn.Module):
    """
    Input:  [B, 1, T]  (T ~ 48000)
    Output: logits [B, 1]  (IMPORTANT: no sigmoid here)
    """
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn_pre   = nn.BatchNorm1d(64)
        self.relu     = nn.ReLU(inplace=True)

        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)

        self.pool = nn.AdaptiveAvgPool1d(64)   # -> [B, 64, 64]

        self.gru  = nn.GRU(
            input_size=64, hidden_size=128, num_layers=1,
            batch_first=True, bidirectional=True
        )

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.bn_pre(self.conv_pre(x)))  # [B,64,T]
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.pool(x)           # [B,64,64]
        x = x.transpose(1, 2)      # [B,64,64] -> [B,seq=64,feat=64]
        out, _ = self.gru(x)       # [B,64,256]
        x = out[:, -1, :]          # [B,256]

        x = self.fc1(x)
        x = self.fc2(x)            # logits
        return x
