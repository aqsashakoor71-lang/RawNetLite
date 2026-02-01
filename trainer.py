import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from RawNetLite import RawNetLite
from torch.utils.data import Dataset, random_split, DataLoader

# ------------------------------
# PARAMETERS
# ------------------------------
BATCH_SIZE = 16
EPOCHS = 20    # quick test; later 10/20
LEARNING_RATE = 1e-4
SEED = 42

# waveform length (3 sec @16k = 48000)
TARGET_LEN = 48000

# ------------------------------
# PATHS
# ------------------------------
MODEL_ROOT = os.path.join(os.getcwd(), "models")
MODEL_NAME = "base_RawNetLite_ASV2019.pt"

ASV2019_BASE = "/kaggle/input/asvpoof-2019-dataset"

# ------------------------------
# SIMPLE METRICS (no sklearn)
# ------------------------------
def simple_accuracy(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    return float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0


def simple_f1(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall + 1e-8))


def best_threshold_by_f1(scores, labels, steps=200):
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    lo, hi = float(scores.min()), float(scores.max())
    thrs = np.linspace(lo, hi, steps)

    best_thr, best_f1v = 0.5, -1.0
    for t in thrs:
        preds = (scores > t).astype(np.int32)
        f1v = simple_f1(labels, preds)
        if f1v > best_f1v:
            best_f1v = f1v
            best_thr = float(t)
    return best_thr, float(best_f1v)


def confusion_counts(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


# ------------------------------
# ✅ ASVspoof2019 LA Dataset (official protocols + flac)
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
            raise ValueError("ASV2019LADataset empty. Check paths.")

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
# DATASET LOADING
# ------------------------------
def load_dataset():
    # ✅ Official ASVspoof2019-LA train split only (then we random split into train/val/test)
    dataset = ASV2019LADataset(ASV2019_BASE, split="train", target_len=TARGET_LEN)
    return dataset


def count_labels(subset):
    ys = [subset[i][1] for i in range(len(subset))]
    ys = np.array(ys, dtype=np.int32)
    return int((ys == 0).sum()), int((ys == 1).sum())


# ------------------------------
# TRAIN
# ------------------------------
def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(MODEL_ROOT, exist_ok=True)

    dataset = load_dataset()
    dataset_size = len(dataset)
    print(f"[INFO] Total samples in dataset: {dataset_size}")

    # Split 80/10/10
    train_len = int(0.8 * dataset_size)
    val_len = int(0.1 * dataset_size)
    test_len = dataset_size - train_len - val_len
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    tr_r, tr_f = count_labels(train_set)
    va_r, va_f = count_labels(val_set)
    te_r, te_f = count_labels(test_set)
    print(f"[INFO] Train bonafide={tr_r} spoof={tr_f}")
    print(f"[INFO] Val   bonafide={va_r} spoof={va_f}")
    print(f"[INFO] Test  bonafide={te_r} spoof={te_f}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = RawNetLite().to(device)

    # ✅ RawNetLite currently returns sigmoid(prob) -> use BCELoss (NOT logits loss)
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1.0
    best_thr = 0.5
    save_path = os.path.join(MODEL_ROOT, MODEL_NAME)

    for epoch in range(EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0

        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            waveforms = waveforms.to(device).float()    # [B,1,T]
            labels = labels.to(device).float()          # [B]
            labels = labels.view(-1, 1)                 # [B,1]

            probs = model(waveforms)                    # [B,1] (already sigmoid)
            loss = criterion(probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss / max(1,len(train_loader)):.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        scores, y_true = [], []

        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                waveforms = waveforms.to(device).float()
                labels = labels.to(device).float()

                probs = model(waveforms).view(-1)  # [B]
                scores.extend(probs.cpu().numpy().tolist())
                y_true.extend(labels.cpu().numpy().tolist())

        thr, _ = best_threshold_by_f1(scores, y_true)
        preds = (np.array(scores) > thr).astype(np.int32)
        acc = simple_accuracy(y_true, preds)
        f1v = simple_f1(y_true, preds)
        tn, fp, fn, tp = confusion_counts(y_true, preds)

        print(f"Validation thr={thr:.4f} | Acc={acc:.4f} | F1={f1v:.4f} | TN={tn} FP={fp} FN={fn} TP={tp}")

        if f1v > best_f1:
            best_f1 = f1v
            best_thr = thr
            torch.save({"model_state_dict": model.state_dict(),
                        "best_thr": best_thr,
                        "best_f1": best_f1}, save_path)
            print(f"[INFO] Saved best model at epoch {epoch+1} with F1={best_f1:.4f} thr={best_thr:.4f}")

    # ---------- TEST ----------
    print("\n[INFO] Evaluation on test set with best saved model:")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    best_thr = float(ckpt.get("best_thr", 0.5))
    model.eval()

    scores, y_true = [], []
    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Testing"):
            waveforms = waveforms.to(device).float()
            labels = labels.to(device).float()
            probs = model(waveforms).view(-1)
            scores.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    preds = (np.array(scores) > best_thr).astype(np.int32)
    acc = simple_accuracy(y_true, preds)
    f1v = simple_f1(y_true, preds)
    tn, fp, fn, tp = confusion_counts(y_true, preds)

    print("\n[TEST RESULTS] on ASVspoof2019-LA (train split random-split)")
    print(f"Best thr(from val): {best_thr:.4f}")
    print(f"Test Accuracy: {acc:.4f} - Test F1: {f1v:.4f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    # class-wise
    print("\nSimple classification report:")
    y_true_np = np.array(y_true, dtype=np.int32)
    y_pred_np = np.array(preds, dtype=np.int32)
    for cls in [0, 1]:
        tp_c = int(((y_true_np == cls) & (y_pred_np == cls)).sum())
        fp_c = int(((y_true_np != cls) & (y_pred_np == cls)).sum())
        fn_c = int(((y_true_np == cls) & (y_pred_np != cls)).sum())
        prec = tp_c / (tp_c + fp_c + 1e-8)
        rec = tp_c / (tp_c + fn_c + 1e-8)
        f1c = 2 * prec * rec / (prec + rec + 1e-8)
        sup = int((y_true_np == cls).sum())
        print(f"class {cls}  prec: {prec:0.4f}  rec: {rec:0.4f}  f1: {f1c:0.4f}  support: {sup}")


if __name__ == "__main__":
    train()
