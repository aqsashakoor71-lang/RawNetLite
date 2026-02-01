import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from focal_loss import FocalLoss
from RawNetLite import RawNetLite
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ------------------------------
# PARAMETERS
# ------------------------------
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 1e-4
SEED = 42

# Raw waveform length (3 seconds @ 16kHz)
TARGET_LEN = 48000

# Loss: "focal" or "bce"
LOSS = "focal"  # or "bce"

# ------------------------------
# PATHS (KAGGLE)
# ------------------------------
ASV2019_BASE = "/kaggle/input/asvpoof-2019-dataset"  # ✅ your Kaggle dataset path

MODEL_ROOT = os.path.join(os.getcwd(), "models")
MODEL_NAME = "RawNetLite_ASV2019_LA.pt"

# ------------------------------
# ASVspoof2019 LA Dataset (official protocols + flac)
# ------------------------------
try:
    import soundfile as sf
except Exception:
    sf = None


class ASV2019LADataset(Dataset):
    """
    Loads ASVspoof2019 LA from official CM protocol (bonafide/spoof).
    Returns:
      x: torch.FloatTensor [1, T]
      y: int  (0=bonafide, 1=spoof)
    """
    def __init__(self, base_root, split="train", target_len=48000):
        if sf is None:
            raise ImportError("soundfile missing. Run: !pip -q install soundfile")

        self.base_root = os.path.join(base_root, "LA", "LA")
        self.split = split
        self.target_len = target_len

        proto_dir = os.path.join(self.base_root, "ASVspoof2019_LA_cm_protocols")

        if split == "train":
            proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt")
            audio_dir = os.path.join(self.base_root, "ASVspoof2019_LA_train", "flac")
        elif split == "dev":
            proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
            audio_dir = os.path.join(self.base_root, "ASVspoof2019_LA_dev", "flac")
        elif split == "eval":
            proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
            audio_dir = os.path.join(self.base_root, "ASVspoof2019_LA_eval", "flac")
        else:
            raise ValueError("split must be train/dev/eval")

        if not os.path.exists(proto):
            raise FileNotFoundError(proto)
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(audio_dir)

        self.items = []
        with open(proto, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                utt_id = parts[1]
                label = parts[-1]  # bonafide/spoof
                y = 0 if label == "bonafide" else 1
                path = os.path.join(audio_dir, utt_id + ".flac")
                if os.path.exists(path):
                    self.items.append((path, y))

        print(f"[INFO] Using ASV2019LADataset split={split} | samples={len(self.items)}")
        if len(self.items) == 0:
            raise ValueError("ASV2019LADataset is empty. Check ASV2019_BASE path.")

    def _fix_len(self, x):
        T = self.target_len
        if len(x) == T:
            return x
        if len(x) > T:
            return x[:T]
        pad = T - len(x)
        return np.pad(x, (0, pad), mode="constant")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        x = self._fix_len(x)
        x = torch.tensor(x).unsqueeze(0)  # [1, T]
        return x, int(y)


# ------------------------------
# DATASET LOADING (ASV19 only)
# ------------------------------
def load_dataset():
    # ✅ training only on official train split
    return ASV2019LADataset(ASV2019_BASE, split="train", target_len=TARGET_LEN)


# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(MODEL_ROOT, exist_ok=True)

    dataset = load_dataset()

    # 80/10/10 split
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    model = RawNetLite().to(device)

    # ✅ RawNetLite.py returns sigmoid(probabilities) -> loss must take probs
    if LOSS == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif LOSS == "bce":
        criterion = nn.BCELoss()
    else:
        raise ValueError("Invalid loss function. Choose 'focal' or 'bce'.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0
    best_path = os.path.join(MODEL_ROOT, MODEL_NAME)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            waveforms = waveforms.to(device).float()                # [B,1,T]
            labels = labels.float().unsqueeze(1).to(device)         # [B,1]

            outputs = model(waveforms)                              # [B,1] probs (sigmoid)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"\nTrain Loss: {running_loss / max(1, len(train_loader)):.4f}")

        # ---------------- VALIDATION ----------------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                waveforms = waveforms.to(device).float()
                outputs = model(waveforms)                          # [B,1] probs
                preds = (outputs > 0.5).int().cpu().numpy().reshape(-1)

                y_true.extend(labels.numpy().reshape(-1))
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Validation Accuracy: {acc:.4f} - F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved best model at epoch {epoch+1} with F1 = {f1:.4f}")

    # ---------------- TEST ----------------
    print("\n[INFO] Evaluation on test set (best checkpoint):")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Testing"):
            waveforms = waveforms.to(device).float()
            outputs = model(waveforms)
            preds = (outputs > 0.5).int().cpu().numpy().reshape(-1)

            y_true.extend(labels.numpy().reshape(-1))
            y_pred.extend(preds)

    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    train()
