import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score

from RawNetLite import RawNetLite

# ============================================================
# PATHS (CONFIRMED)
# ============================================================
ASV19_LA_ROOT = "/kaggle/input/asvpoof-2019-dataset/LA/LA"

ASV19_CM_PROTO_DIR = os.path.join(ASV19_LA_ROOT, "ASVspoof2019_LA_cm_protocols")
ASV19_TRAIN_AUDIO_DIR = os.path.join(ASV19_LA_ROOT, "ASVspoof2019_LA_train", "flac")
ASV19_DEV_AUDIO_DIR   = os.path.join(ASV19_LA_ROOT, "ASVspoof2019_LA_dev", "flac")

ASV19_TRAIN_PROTO = os.path.join(
    ASV19_CM_PROTO_DIR, "ASVspoof2019.LA.cm.train.trn.txt"
)
ASV19_DEV_PROTO = os.path.join(
    ASV19_CM_PROTO_DIR, "ASVspoof2019.LA.cm.dev.trl.txt"
)

# ============================================================
# TRAINING CONFIG
# ============================================================
SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

SR = 16000
CLIP_LEN = 48000

# ============================================================
# MODEL SAVE (SAFE)
# ============================================================
MODEL_DIR = "/kaggle/working/models"
BACKUP_DIR = "/kaggle/working/backup_models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

MODEL_NAME = "RawNetLite_A3_PreEmp_Attn_ASV19_E20.pt"
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
BACKUP_PATH = os.path.join(BACKUP_DIR, MODEL_NAME)

# ============================================================
# SEED
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# AUDIO UTILS
# ============================================================
def load_audio_safe(path):
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            return None
        return wav.astype(np.float32)
    except Exception:
        return None

def normalize(wav):
    m = np.max(np.abs(wav))
    return wav if m < 1e-8 else wav / m

def pad_trim(wav, L):
    if len(wav) >= L:
        return wav[:L]
    out = np.zeros(L, dtype=np.float32)
    out[:len(wav)] = wav
    return out

# ============================================================
# PROTOCOL PARSER
# ============================================================
def parse_cm_protocol(proto):
    items = []
    with open(proto, "r") as f:
        for line in f:
            parts = line.strip().split()
            fid = parts[1]
            label = parts[-1].lower()
            y = 0 if label == "bonafide" else 1
            items.append((fid, y))
    return items

# ============================================================
# DATASET
# ============================================================
class ASVspoof2019CMDataset(Dataset):
    def __init__(self, audio_dir, proto):
        self.items = []
        for fid, y in parse_cm_protocol(proto):
            path = os.path.join(audio_dir, fid + ".flac")
            if os.path.isfile(path):
                self.items.append((path, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        wav = load_audio_safe(path)
        if wav is None:
            return None
        wav = normalize(wav)
        wav = pad_trim(wav, CLIP_LEN)
        x = torch.from_numpy(wav).unsqueeze(0)
        return x, y

# ============================================================
# COLLATE
# ============================================================
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if len(batch) > 0 else None

# ============================================================
# METRICS
# ============================================================
def compute_eer(scores, labels):
    idx = np.argsort(scores)[::-1]
    scores, labels = scores[idx], labels[idx]
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    fp, fn = 0, P
    eer = 1.0
    for i in range(len(scores)):
        if labels[i] == 1:
            fn -= 1
        else:
            fp += 1
        fpr, fnr = fp / N, fn / P
        eer = min(eer, (fpr + fnr) / 2)
    return eer

# ============================================================
# TRAIN / DEV
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(loader, desc="train"):
        if batch is None:
            continue

        x, y = batch
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)

@torch.no_grad()
def eval_dev(model, loader, device):
    model.eval()
    probs, labels = [], []

    for batch in loader:
        if batch is None:
            continue

        x, y = batch
        x = x.to(device)

        logits = model(x)
        p = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        probs.append(p)
        labels.append(np.array(y))

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    eer = compute_eer(probs, labels)

    return acc, f1, eer

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RawNetLite().to(device)

    train_ds = ASVspoof2019CMDataset(
        ASV19_TRAIN_AUDIO_DIR, ASV19_TRAIN_PROTO
    )
    dev_ds = ASVspoof2019CMDataset(
        ASV19_DEV_AUDIO_DIR, ASV19_DEV_PROTO
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=safe_collate,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=safe_collate,
        pin_memory=True
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_eer = 1.0

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        acc, f1, eer = eval_dev(model, dev_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={loss:.4f} | "
            f"acc={acc:.4f} | "
            f"f1={f1:.4f} | "
            f"dev_EER={eer:.4f}"
        )

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), SAVE_PATH)
            torch.save(model.state_dict(), BACKUP_PATH)
            print("✅ BEST MODEL SAVED")

    print("\n🎉 TRAINING FINISHED")
    print("Saved model:")
    print(SAVE_PATH)
    print(BACKUP_PATH)

if __name__ == "__main__":
    main()
