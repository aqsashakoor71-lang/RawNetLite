import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import soundfile as sf

from RawNetLite import RawNetLite

# -------------------------
# PATHS (as per your request)
# -------------------------
ASV19_LA_ROOT = "/kaggle/input/asvpoof-2019-dataset/LA"
ASV21_LA_EVAL_ROOT = "/kaggle/input/avsspoof-2021/ASVspoof2021_LA_eval"

# -------------------------
# TRAINING CONFIG
# -------------------------
SEED = 42
BATCH_SIZE = 16
EPOCHS = 35
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

SR = 16000
CLIP_LEN = 48000  # 3 seconds @ 16k

SAVE_DIR = "/kaggle/working/checkpoints"
SAVE_PATH = os.path.join(SAVE_DIR, "rawnetlite_asv19LA_best.pt")

# -------------------------
# REPRODUCIBILITY
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# AUDIO
# -------------------------
def load_audio(path: str) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    # ASVspoof is usually 16k. If not 16k, we should resample.
    # Minimal safe: if mismatch, we throw a clear error (better than silent wrong).
    if sr != SR:
        raise ValueError(f"Sample rate mismatch: {sr} != {SR} for file: {path}")
    return wav

def normalize(wav: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(wav))
    if m < 1e-8:
        return wav
    return wav / m

def pad_trim(wav: np.ndarray, L: int) -> np.ndarray:
    if len(wav) >= L:
        return wav[:L]
    out = np.zeros(L, dtype=np.float32)
    out[:len(wav)] = wav
    return out

# -------------------------
# PROTOCOL HELPERS
# -------------------------
def find_first_file(root: str, pattern: str) -> str:
    """Find first file matching regex pattern in root (recursive)."""
    rx = re.compile(pattern)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if rx.search(fn):
                return os.path.join(dirpath, fn)
    return ""

def parse_protocol_lines(proto_path: str):
    """
    Returns list of tuples: (file_id, label or None)
    Accepts common ASVspoof formats.
    """
    items = []
    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            file_id = parts[1]

            # Usually last token is cm_label: bonafide/spoof
            last = parts[-1].lower()
            if last in ("bonafide", "spoof"):
                y = 0 if last == "bonafide" else 1
            else:
                # Some eval protocols may not include labels
                y = None

            items.append((file_id, y))
    return items

# -------------------------
# DATASETS
# -------------------------
class ASVspoof2019LADataset(Dataset):
    """
    Expects ASVspoof2019 LA structure under:
    /kaggle/input/asvpoof-2019-dataset/LA/
        ASVspoof2019_LA_train/flac/*.flac
        ASVspoof2019_LA_train/protocols/ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019_LA_dev/flac/*.flac
        ASVspoof2019_LA_dev/protocols/ASVspoof2019.LA.cm.dev.trl.txt
    """
    def __init__(self, la_root: str, split: str):
        super().__init__()
        self.la_root = la_root
        self.split = split  # "train" or "dev"
        base = f"ASVspoof2019_LA_{split}"
        self.audio_dir = os.path.join(la_root, base, "flac")
        proto_dir = os.path.join(la_root, base, "protocols")

        if split == "train":
            proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt")
        elif split == "dev":
            proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
        else:
            raise ValueError("split must be 'train' or 'dev'")

        if not os.path.isfile(proto):
            raise FileNotFoundError(f"Protocol not found: {proto}")
        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"Audio folder not found: {self.audio_dir}")

        raw = parse_protocol_lines(proto)
        self.items = []
        for fid, y in raw:
            if y is None:
                continue
            path = os.path.join(self.audio_dir, fid + ".flac")
            self.items.append((path, y))

        if len(self.items) == 0:
            raise RuntimeError(f"No samples found for {split}. Check paths/protocol.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        wav = load_audio(path)
        wav = normalize(wav)
        wav = pad_trim(wav, CLIP_LEN)
        x = torch.from_numpy(wav).unsqueeze(0)  # [1, L]
        return x, y

class ASVspoof2021LAEvalDataset(Dataset):
    """
    Works with eval folder:
    /kaggle/input/avsspoof-2021/ASVspoof2021_LA_eval
    Tries to locate:
      - flac/ OR audio/ folder (we search)
      - a protocol .txt containing eval labels if present
    If labels are missing, we still output scores (no EER).
    """
    def __init__(self, eval_root: str):
        super().__init__()
        self.eval_root = eval_root

        # Find audio folder (common: "flac")
        flac_dir = os.path.join(eval_root, "flac")
        if os.path.isdir(flac_dir):
            self.audio_dir = flac_dir
        else:
            # fallback: search for a folder containing many .flac
            self.audio_dir = ""
            best = (0, "")
            for dirpath, _, filenames in os.walk(eval_root):
                n = sum(fn.lower().endswith(".flac") for fn in filenames)
                if n > best[0]:
                    best = (n, dirpath)
            if best[0] > 0:
                self.audio_dir = best[1]
            else:
                raise FileNotFoundError(f"Could not find any .flac files under: {eval_root}")

        # Find protocol txt (eval)
        # We search for something containing "eval" and ".txt"
        proto = find_first_file(eval_root, r"eval.*\.txt$")
        if proto == "":
            # fallback: any txt in eval_root
            proto = find_first_file(eval_root, r"\.txt$")
        self.proto_path = proto  # may be ""

        self.items = []
        self.has_labels = False

        if self.proto_path and os.path.isfile(self.proto_path):
            raw = parse_protocol_lines(self.proto_path)
            for fid, y in raw:
                path = os.path.join(self.audio_dir, fid + ".flac")
                self.items.append((path, y))
                if y is not None:
                    self.has_labels = True
        else:
            # No protocol: just list all flac files
            for fn in sorted(os.listdir(self.audio_dir)):
                if fn.lower().endswith(".flac"):
                    self.items.append((os.path.join(self.audio_dir, fn), None))

        if len(self.items) == 0:
            raise RuntimeError("No eval items found. Check eval root path.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        wav = load_audio(path)
        wav = normalize(wav)
        wav = pad_trim(wav, CLIP_LEN)
        x = torch.from_numpy(wav).unsqueeze(0)
        return x, y, os.path.basename(path)

# -------------------------
# METRIC: EER
# -------------------------
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    # labels: 1=spoof, 0=bonafide
    idx = np.argsort(scores)[::-1]
    scores = scores[idx]
    labels = labels[idx]

    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    if P == 0 or N == 0:
        return float("nan")

    fp = 0
    fn = P
    best = 1.0
    eer = 1.0

    for i in range(len(scores)):
        if labels[i] == 1:
            fn -= 1
        else:
            fp += 1
        fpr = fp / N
        fnr = fn / P
        d = abs(fpr - fnr)
        if d < best:
            best = d
            eer = (fpr + fnr) / 2.0
    return float(eer)

# -------------------------
# SANITY (anti-collapse)
# -------------------------
@torch.no_grad()
def sanity_check(model, loader, device):
    batch = next(iter(loader))
    x, y = batch
    x = x.to(device)
    y = torch.tensor(y).to(device)

    print("Sanity | x shape:", tuple(x.shape))
    print("Sanity | label counts:", torch.unique(y, return_counts=True))

    xmin, xmax = x.min().item(), x.max().item()
    xmean, xstd = x.mean().item(), x.std().item()
    print(f"Sanity | x min/max={xmin:.4f}/{xmax:.4f} mean/std={xmean:.4f}/{xstd:.4f}")

    logits = model(x)
    lmin, lmax = logits.min().item(), logits.max().item()
    lmean, lstd = logits.mean().item(), logits.std().item()
    print(f"Sanity | logits min/max={lmin:.4f}/{lmax:.4f} mean/std={lmean:.4f}/{lstd:.6f}")

    # If logits std is extremely tiny at init, it's not fatal, but warn
    if lstd < 1e-6:
        print("⚠️ Warning: logits std extremely small at init. If it stays small after training steps -> collapse risk.")

def grad_check(model):
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.abs().mean().item())
            count += 1
    return total / max(count, 1)

# -------------------------
# TRAIN + EVAL
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        g = grad_check(model)
        if not np.isfinite(g) or g == 0.0:
            print("⚠️ Gradient looks bad (zero or NaN). This can cause collapse. Check data/labels/loss.")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)

@torch.no_grad()
def eval_eer_on_loader(model, loader, device):
    model.eval()
    scores = []
    labels = []
    for x, y in tqdm(loader, desc="dev-eval", leave=False):
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        scores.append(prob)
        labels.append(np.array(y))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    return compute_eer(scores, labels)

@torch.no_grad()
def eval_asv21(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()

    out_scores = []
    out_labels = []
    out_files = []

    for x, y, fname in tqdm(loader, desc="ASV21-eval", leave=False):
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        out_scores.append(prob)
        out_files.extend(list(fname))

        if y[0] is not None:
            # y is a list with possible None; convert carefully
            yy = []
            for v in y:
                if v is None:
                    yy.append(-1)
                else:
                    yy.append(int(v))
            out_labels.append(np.array(yy))

    scores = np.concatenate(out_scores)

    # Save score file (always)
    score_path = "/kaggle/working/asv21_eval_scores.txt"
    with open(score_path, "w", encoding="utf-8") as f:
        for fn, sc in zip(out_files, scores):
            f.write(f"{fn}\t{sc:.8f}\n")
    print("✅ Saved scores:", score_path)

    # If labels exist -> EER
    if dataset.has_labels and len(out_labels) > 0:
        labels = np.concatenate(out_labels)
        # Remove unlabeled (-1)
        mask = labels >= 0
        eer = compute_eer(scores[mask], labels[mask])
        print(f"✅ ASVspoof2021 LA eval EER: {eer:.4f}")
    else:
        print("ℹ️ ASVspoof2021 eval protocol labels not found. Scores saved; EER not computed.")

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNetLite().to(device)

    # Data
    train_ds = ASVspoof2019LADataset(ASV19_LA_ROOT, "train")
    dev_ds   = ASVspoof2019LADataset(ASV19_LA_ROOT, "dev")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Anti-collapse sanity before training
    sanity_check(model, train_loader, device)

    # Loss (paper-aligned, collapse-safe)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_eer = 1e9

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_eer = eval_eer_on_loader(model, dev_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | dev_EER={dev_eer:.4f}")

        # Collapse detector: if model outputs almost constant on dev, warn strongly
        # (We do a quick probe on 1 batch)
        with torch.no_grad():
            xb, yb = next(iter(dev_loader))
            xb = xb.to(device)
            logits = model(xb)
            std = logits.std().item()
            if std < 1e-4:
                print("🚨 COLLAPSE WARNING: logits std is extremely small (<1e-4).")
                print("   Check: labels distribution, data loading, loss mismatch, frozen params, or audio zeros.")

        if dev_eer < best_eer:
            best_eer = dev_eer
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Saved best model: {SAVE_PATH} (best_dev_EER={best_eer:.4f})")

    # Load best + evaluate on ASVspoof2021 eval
    model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
    model.to(device)

    asv21_eval_ds = ASVspoof2021LAEvalDataset(ASV21_LA_EVAL_ROOT)
    print("\n--- ASVspoof2021 LA eval info ---")
    print("Audio dir:", asv21_eval_ds.audio_dir)
    print("Protocol :", asv21_eval_ds.proto_path if asv21_eval_ds.proto_path else "(not found)")
    print("Has labels:", asv21_eval_ds.has_labels)

    eval_asv21(model, asv21_eval_ds, device)

if __name__ == "__main__":
    main()
