import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# ✅ ADDED: Squeeze-and-Excitation Block (SE)
# ============================
class SEBlock(nn.Module):
    """
    SE Block for 1D CNN features.
    Input:  [B, C, T]
    Output: [B, C, T]  (same shape)

    What it does:
    - Squeeze: time-wise global average -> [B, C]
    - Excite: learn channel weights -> [B, C] (0..1)
    - Scale: multiply original features channel-wise
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        # NOTE: Ensure hidden size is at least 1 (safe even if channels < reduction)
        hidden = max(1, channels // reduction)

        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.size()

        # SQUEEZE: Global average pooling over time axis
        s = x.mean(dim=2)              # [B, C]

        # EXCITATION: Two FC layers to produce channel gates
        s = F.relu(self.fc1(s))        # [B, hidden]
        s = torch.sigmoid(self.fc2(s)) # [B, C]

        # SCALE: reshape to [B, C, 1] and multiply
        s = s.view(b, c, 1)
        return x * s


# Residual block base
class ResBlock(nn.Module):
    """
    A 1D convolutional residual block for processing sequential data.

    Each block consists of two convolutional layers with BatchNorm and ReLU activation.
    The input is added to the output (residual connection), enabling gradient flow and improving convergence.

    ✅ Modification:
    - Added SEBlock to learn channel importance (helps cross-domain robustness)
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

        # ✅ ADDED: SE block (channel re-weighting)
        self.se = SEBlock(channels, reduction=16)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # Old return (BEFORE SE) — keep as comment (as requested)
        # return self.relu(x + residual)

        # Residual connection
        x = self.relu(x + residual)

        # ✅ Apply SE after residual (common stable placement)
        x = self.se(x)

        return x


# RawNetLite model
class RawNetLite(nn.Module):
    """
    RawNetLite: A lightweight end-to-end architecture for audio deepfake detection.

    ✅ Modification:
    - ResBlocks now include SE channel attention internally.
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

        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=1,
                          batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

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
        x = self.fc2(x)                                  # [B, 1]
        return torch.sigmoid(x)
