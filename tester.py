import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- install soundfile if needed ----
try:
    import soundfile as sf
except Exception:
    !pip -q install soundfile
    import soundfile as sf

from RawNetLite import RawNetLite

# -----------------------
# PATHS
# -----------------------
CKPT_PATH = "/kaggle/working/RawNetLite/models/SE_RawNetLite_ASV2019.pt"  # from patched trainer.py
META_2021 = "/kaggle/input/avsspoof-2021/LA-keys-full/keys/LA/CM/trial_metadata.txt"
AUDIO_2021_DIR = "/kaggle/input/avsspoof-2021/ASVspoof2021_LA_eval/flac"

TARGET_LEN = 48000  # 3 seconds @16k
BATCH_SIZE = 16

print("CKPT exists:", os.path.exists(CKPT_PATH), CKPT_PATH)
print("META exists:", os.path.exists(META_2021), META_2021)
print("AUDIO dir exists:", os.path.exists(AUDIO_2021_DIR), AUDIO_2021_DIR)

# -----------------------
# Metrics (no sklearn)
# -----------------------
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
    return float(2 * precision * recall / (precision + recall + 1e-8)) if (precision+recall)>0 else 0.0

def confusion_counts(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp

def compute_eer(labels, scores):
    """
    EER without sklearn:
    - sort by score
    - sweep threshold
    - find point where FPR ~= FNR
    """
    labels = np.array(labels, dtype=np.int32)
    scores = np.array(scores, dtype=np.float64)

    # sort by score descending
    idx = np.argsort(scores)[::-1]
    labels = labels[idx]
    scores = scores[idx]

    P = (labels == 1).sum()
    N = (labels == 0).sum()
    if P == 0 or N == 0:
        return np.nan, np.nan

    tp = 0
    fp = 0

    best_eer = 1.0
    best_thr = scores[0]

    # thresholds at each unique score
    for i in range(len(scores)):
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1

        # when score changes, evaluate at that threshold
        if i == len(scores)-1 or scores[i] != scores[i+1]:
            tpr = tp / (P + 1e-12)
            fpr = fp / (N + 1e-12)
            fnr = 1 - tpr
            diff = abs(fpr - fnr)
            eer = (fpr + fnr) / 2
            if diff < best_eer:
                best_eer = diff
                best_thr = scores[i]
                best_val = eer

    return float(best_val), float(best_thr)

# -----------------------
# Dataset for 2021 eval
# -----------------------
class ASV2021LAEvalDataset(Dataset):
    """
    trial_metadata format sample:
    LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
             ^ file id is 2nd column (LA_E_...)
                              label is 6th column (bonafide/spoof)
    """
    def __init__(self, meta_path, audio_dir, target_len=48000):
        self.audio_dir = audio_dir
        self.target_len = target_len
        self.items = []

        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 6:
                    continue
                utt = p[1]
                lab = p[5]  # bonafide/spoof
                y = 0 if lab == "bonafide" else 1
                path = os.path.join(audio_dir, utt + ".flac")
                if os.path.exists(path):
                    self.items.append((path, y))

        print(f"[INFO] 2021-LA eval loaded: {len(self.items)} samples")

    def _fix_len(self, x):
        T = self.target_len
        if len(x) == T:
            return x
        if len(x) > T:
            return x[:T]
        return np.pad(x, (0, T - len(x)), mode="constant")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        x = self._fix_len(x)
        x = torch.tensor(x).unsqueeze(0)  # [1,T]
        return x, y

# -----------------------
# Load model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RawNetLite().to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    best_thr_from_val = float(ckpt.get("best_thr", 0.5))
else:
    # in case it was saved directly as state_dict
    model.load_state_dict(ckpt, strict=False)
    best_thr_from_val = 0.5

model.eval()
print("[INFO] Loaded model. best_thr_from_val:", best_thr_from_val)

# -----------------------
# Run evaluation
# -----------------------
ds = ASV2021LAEvalDataset(META_2021, AUDIO_2021_DIR, target_len=TARGET_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

scores = []
y_true = []

with torch.no_grad():
    for x, y in tqdm(dl, desc="ASV2021 eval"):
        x = x.to(device).float()
        out = model(x).view(-1)  # already sigmoid probs
        scores.extend(out.detach().cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())

# decision using fixed threshold (val threshold from 2019 train)
preds_fixed = (np.array(scores) > best_thr_from_val).astype(np.int32)
acc_fixed = simple_accuracy(y_true, preds_fixed)
f1_fixed = simple_f1(y_true, preds_fixed)
tn, fp, fn, tp = confusion_counts(y_true, preds_fixed)
eer, eer_thr = compute_eer(y_true, scores)

print("\n=== ASVspoof2021-LA eval (Cross-domain) ===")
print(f"Samples: {len(y_true)}")
print(f"Fixed threshold (from 2019-val): {best_thr_from_val:.4f}")
print(f"Acc: {acc_fixed:.4f} | F1: {f1_fixed:.4f}")
print(f"Confusion [TN FP FN TP]: {tn} {fp} {fn} {tp}")
print(f"EER: {eer:.4f} @ thr={eer_thr:.4f}")
