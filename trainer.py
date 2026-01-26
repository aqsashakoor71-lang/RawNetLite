import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

from focal_loss import FocalLoss
from RawNetLite import RawNetLite

# ❌ OLD PROCESSED DATASET (NOT USED)
# from FOR_dataset import FakeOrRealTestDataset

# ✅ USE DIRECT ASVSPOOF DATASET
from AVSpoof_dataset import AVSpoofTestDataset

from torch.utils.data import random_split, DataLoader

# ======================================================
# PARAMETERS (FAST + SAFE FOR KAGGLE)
# ======================================================
BATCH_SIZE = 8
EPOCHS = 1                 # Stage-A demo (increase later)
LEARNING_RATE = 1e-4
SEED = 42

LOSS = "focal"             # focal works better for imbalance

# ======================================================
# PATHS
# ======================================================
MODEL_ROOT = os.path.join(os.getcwd(), "models")
MODEL_NAME = "rawnetlite_multitask_cm_quality.pt"

# ✅ ASVspoof2019-LA paths (DIRECT FLAC)
ASV19_AUDIO_DIR = "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac"
ASV19_PROTO = "/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# ======================================================
# SIMPLE METRICS (NO SKLEARN)
# ======================================================
def simple_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())


def simple_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


# ======================================================
# QUALITY DEGRADATION (CHEAP & FAST)
# ======================================================
def degrade_waveform(wav):
    # simulates codec/channel loss
    x = F.avg_pool1d(wav, kernel_size=4, stride=4)
    x = x.repeat_interleave(4, dim=-1)
    return x[..., : wav.size(-1)]


def make_quality_labels(bs, device):
    # 0 = clean, 1 = degraded
    return (torch.rand(bs, device=device) < 0.5).long()


# ======================================================
# DATASET
# ======================================================
def load_dataset():
    print("[INFO] Loading ASVspoof2019-LA directly from flac")

    dataset = AVSpoofTestDataset(
        protocol_file=ASV19_PROTO,
        audio_dir=ASV19_AUDIO_DIR,
        max_len=64000
    )
    return dataset


# ======================================================
# TRAINING
# ======================================================
def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(MODEL_ROOT, exist_ok=True)

    dataset = load_dataset()
    print("[INFO] Total samples:", len(dataset))
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty — check paths")

    # split
    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model = RawNetLite().to(device)

    criterion_cm = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0

    # ===========================
    # EPOCH LOOP
    # ===========================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for wav, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            wav = wav.to(device)
            labels = labels.to(device).float()

            q_labels = make_quality_labels(wav.size(0), device)
            wav_deg = degrade_waveform(wav)
            mask = (q_labels == 1).view(-1, 1, 1)
            wav_used = torch.where(mask, wav_deg, wav)

            cm_logits, q_logits = model(wav_used)

            cm_prob = torch.sigmoid(cm_logits).squeeze()
            loss_cm = criterion_cm(cm_prob, labels)
            loss_q = F.cross_entropy(q_logits, q_labels)

            loss = loss_cm + 0.3 * loss_q

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Train loss: {total_loss / len(train_loader):.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for wav, labels in val_loader:
                wav = wav.to(device)
                labels = labels.to(device)

                cm_logits, _ = model(wav)
                preds = (torch.sigmoid(cm_logits).squeeze() > 0.5).float()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = simple_accuracy(y_true, y_pred)
        f1 = simple_f1(y_true, y_pred)
        print(f"Val Acc: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(MODEL_ROOT, MODEL_NAME))
            print("✅ Best model saved")

    # ===========================
    # TEST
    # ===========================
    print("\n[TEST]")
    model.load_state_dict(torch.load(os.path.join(MODEL_ROOT, MODEL_NAME), map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for wav, labels in test_loader:
            wav = wav.to(device)
            labels = labels.to(device)

            cm_logits, _ = model(wav)
            preds = (torch.sigmoid(cm_logits).squeeze() > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Test Acc:", simple_accuracy(y_true, y_pred))
    print("Test F1 :", simple_f1(y_true, y_pred))


if __name__ == "__main__":
    train()
