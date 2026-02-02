import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import soundfile as sf

from RawNetLite import RawNetLite

# ============================================================
# PATHS (YOUR VERIFIED KAGGLE PATHS)
# ============================================================
ASV19_LA_ROOT = "/kaggle/input/asvpoof-2019-dataset/LA/LA"

ASV19_CM_PROTO_DIR = os.path.join(ASV19_LA_ROOT, "ASVspoof2019_LA_cm_protocols")
ASV19_TRAIN_AUDIO_DIR = os.path.join(ASV19_LA_ROOT, "ASVspoof2019_LA_train", "flac")
ASV19_DEV_AUDIO_DIR   = os.path.join(ASV19_LA_ROOT, "ASVspoof2019_LA_dev", "flac")
ASV19_EVAL_AUDIO_DIR  = os.path.join(ASV19_LA_ROOT, "ASVspoof2019_LA_eval", "flac")

ASV19_TRAIN_PROTO = os.path.join(ASV19_CM_PROTO_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
ASV19_DEV_PROTO   = os.path.join(ASV19_CM_PROTO_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")
ASV19_EVAL_PROTO  = os.path.join(ASV19_CM_PROTO_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")

# ASVspoof2021 eval root
ASV21_LA_EVAL_ROOT = "/kaggle/input/avsspoof-2021/ASVspoof2021_LA_eval"

# ============================================================
# TRAINING CONFIG (SAFE)
# ============================================================
SEED = 42
BATCH_SIZE = 16
EPOCHS = 25          # <-- Kaggle free GPU friendly (you can set 20-30)
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

SR = 16000
CLIP_LEN = 48000  # 3 sec @ 16k

# ============================================================
# UNIQUE SAVE NAME (NO OVERWRITE)
# ============================================================
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = f"RawNetLite_ASV19LA_to_ASV21LA_seed{SEED}_lr{LR}_len{CLIP_LEN}_epochs{EPOCHS}_v3.pt"
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

ASV19_SCORE_PATH = os.path.join(os.getcwd(), f"asv19_eval_scores_seed{SEED}_v3.txt")
ASV21_SCORE_PATH = os.path.join(os.getcwd(), f"asv21_eval_scores_seed{SEED}_v3.txt")

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# AUDIO UTILS (CORRUPTION-SAFE)
# ============================================================
def load_audio_safe(path: str):
    """
    Returns np.float32 wav if ok else None (corrupted/unreadable).
    """
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != SR:
            # safer to fail this file only
            return None
        return wav
    except Exception:
        return None

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

# ============================================================
# PROTOCOL PARSER (ASVspoof2019 CM)
# ============================================================
def parse_asv19_cm_protocol(proto_path: str):
    """
    label: bonafide/spoof
    y: 0=bonafide, 1=spoof
    """
    items = []
    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            file_id = parts[1]
            label = parts[-1].lower()
            if label not in ("bonafide", "spoof"):
                continue
            y = 0 if label == "bonafide" else 1
            items.append((file_id, y))
    return items

# ============================================================
# DATASETS
# ============================================================
class ASVspoof2019CMDataset(Dataset):
    """
    Train/Dev/Eval for ASVspoof2019 LA using CM protocols.
    Returns: x, y, fid
    NOTE: If a specific file is unreadable (rare), we skip via None + safe_collate.
    """
    def __init__(self, audio_dir: str, proto_path: str):
        super().__init__()
        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
        if not os.path.isfile(proto_path):
            raise FileNotFoundError(f"Protocol not found: {proto_path}")

        raw = parse_asv19_cm_protocol(proto_path)
        self.items = []
        missing = 0
        for fid, y in raw:
            path = os.path.join(audio_dir, fid + ".flac")
            if os.path.isfile(path):
                self.items.append((path, y, fid))
            else:
                missing += 1

        if len(self.items) == 0:
            raise RuntimeError(f"No items loaded. Missing files: {missing}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y, fid = self.items[idx]
        wav = load_audio_safe(path)
        if wav is None:
            return None  # will be skipped by safe_collate
        wav = normalize(wav)
        wav = pad_trim(wav, CLIP_LEN)
        x = torch.from_numpy(wav).unsqueeze(0)
        return x, y, fid

class ASVspoof2021LAEvalDataset(Dataset):
    """
    ASVspoof2021 eval:
    - Finds flac dir automatically
    - Tries to find a protocol text; if labels exist we can compute EER, else scores only
    - CORRUPTED FLAC: returns None (skipped by safe_collate)
    """
    def __init__(self, eval_root: str):
        super().__init__()
        self.eval_root = eval_root

        # Find directory that has maximum .flac files
        best = (0, "")
        for dirpath, _, filenames in os.walk(eval_root):
            n = sum(fn.lower().endswith(".flac") for fn in filenames)
            if n > best[0]:
                best = (n, dirpath)
        if best[0] == 0:
            raise FileNotFoundError(f"No .flac found under: {eval_root}")
        self.audio_dir = best[1]

        # Try find a protocol (optional)
        self.proto_path = ""
        for dirpath, _, filenames in os.walk(eval_root):
            for fn in filenames:
                low = fn.lower()
                if low.endswith(".txt") and ("eval" in low or "protocol" in low):
                    self.proto_path = os.path.join(dirpath, fn)
                    break
            if self.proto_path:
                break

        self.items = []
        self.has_labels = False

        if self.proto_path and os.path.isfile(self.proto_path):
            with open(self.proto_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    fid = parts[1]
                    last = parts[-1].lower()
                    if last in ("bonafide", "spoof"):
                        y = 0 if last == "bonafide" else 1
                        self.has_labels = True
                    else:
                        y = None
                    path = os.path.join(self.audio_dir, fid + ".flac")
                    self.items.append((path, y, fid))
        else:
            for fn in sorted(os.listdir(self.audio_dir)):
                if fn.lower().endswith(".flac"):
                    fid = os.path.splitext(fn)[0]
                    self.items.append((os.path.join(self.audio_dir, fn), None, fid))

        if len(self.items) == 0:
            raise RuntimeError("No eval items found in ASVspoof2021 eval dataset.")

        self.skipped = 0  # count corrupted files skipped during __getitem__

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y, fid = self.items[idx]
        wav = load_audio_safe(path)
        if wav is None:
            # corrupted file -> skip
            self.skipped += 1
            return None
        wav = normalize(wav)
        wav = pad_trim(wav, CLIP_LEN)
        x = torch.from_numpy(wav).unsqueeze(0)
        return x, y, fid

# ============================================================
# COLLATE (SKIP None SAMPLES)
# ============================================================
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# ============================================================
# EER
# ============================================================
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
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

# ============================================================
# ANTI-COLLAPSE SANITY
# ============================================================
@torch.no_grad()
def sanity_check(model, loader, device, name=""):
    # find a non-empty batch
    batch = None
    for _ in range(50):
        b = next(iter(loader))
        if b is not None:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"Sanity check failed: all batches None for {name}")

    x, y, _ = batch
    x = x.to(device)
    y = torch.tensor(y).to(device)

    print(f"\nSanity {name} | x shape:", tuple(x.shape))
    print(f"Sanity {name} | label counts:", torch.unique(y, return_counts=True))
    print(f"Sanity {name} | x min/max: {x.min().item():.4f}/{x.max().item():.4f} "
          f"mean/std: {x.mean().item():.4f}/{x.std().item():.4f}")

    logits = model(x)
    print(f"Sanity {name} | logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f} "
          f"mean/std: {logits.mean().item():.4f}/{logits.std().item():.6f}")

def grad_mean_abs(model):
    s = 0.0
    c = 0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.abs().mean().item())
            c += 1
    return s / max(c, 1)

# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        if batch is None:
            continue
        x, y, _ = batch
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        g = grad_mean_abs(model)
        if not np.isfinite(g) or g == 0.0:
            print("⚠️ Gradient looks zero/NaN. Check data/labels/loss.")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)

@torch.no_grad()
def eval_eer(model, loader, device, name=""):
    model.eval()
    scores = []
    labels = []
    for batch in tqdm(loader, desc=f"{name}-eval", leave=False):
        if batch is None:
            continue
        x, y, _ = batch
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        scores.append(prob)
        labels.append(np.array(y))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    return compute_eer(scores, labels)

@torch.no_grad()
def save_scores(model, loader, device, out_path: str, name=""):
    model.eval()
    all_scores = []
    all_ids = []
    kept = 0
    skipped = 0

    for batch in tqdm(loader, desc=f"{name}-scores", leave=False):
        if batch is None:
            skipped += 1
            continue
        x, _, fid = batch
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.append(prob)
        all_ids.extend(list(fid))
        kept += len(fid)

    scores = np.concatenate(all_scores) if len(all_scores) else np.array([], dtype=np.float32)

    with open(out_path, "w", encoding="utf-8") as f:
        for fid, sc in zip(all_ids, scores):
            f.write(f"{fid}\t{sc:.8f}\n")

    print(f"✅ Saved scores ({name}): {out_path}")
    print(f"   kept_samples={kept}, skipped_batches={skipped}")

@torch.no_grad()
def eval_asv21_and_save(model, eval_ds, device, out_path: str):
    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=safe_collate,
    )

    model.eval()
    all_scores = []
    all_ids = []
    all_labels = []
    has_labels = eval_ds.has_labels

    kept = 0
    empty_batches = 0

    for batch in tqdm(loader, desc="ASV21-eval", leave=False):
        if batch is None:
            empty_batches += 1
            continue
        x, y, fid = batch
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        all_scores.append(prob)
        all_ids.extend(list(fid))
        kept += len(fid)

        if has_labels:
            yy = []
            for v in y:
                yy.append(int(v) if v is not None else -1)
            all_labels.append(np.array(yy))

    scores = np.concatenate(all_scores) if len(all_scores) else np.array([], dtype=np.float32)

    with open(out_path, "w", encoding="utf-8") as f:
        for fid, sc in zip(all_ids, scores):
            f.write(f"{fid}\t{sc:.8f}\n")

    print(f"✅ Saved ASV21 scores: {out_path}")
    print(f"   kept_files={kept}, empty_batches={empty_batches}, corrupted_skipped_files={eval_ds.skipped}")

    if has_labels and len(all_labels) > 0 and len(scores) > 0:
        labels = np.concatenate(all_labels)
        mask = labels >= 0
        if np.any(mask):
            eer = compute_eer(scores[mask], labels[mask])
            return eer
    return None

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Fail-fast checks (ASV19 only)
    must_exist = [
        ASV19_TRAIN_AUDIO_DIR, ASV19_DEV_AUDIO_DIR, ASV19_EVAL_AUDIO_DIR,
        ASV19_TRAIN_PROTO, ASV19_DEV_PROTO, ASV19_EVAL_PROTO
    ]
    for p in must_exist:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing path: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNetLite().to(device)

    train_ds = ASVspoof2019CMDataset(ASV19_TRAIN_AUDIO_DIR, ASV19_TRAIN_PROTO)
    dev_ds   = ASVspoof2019CMDataset(ASV19_DEV_AUDIO_DIR, ASV19_DEV_PROTO)
    eval19_ds= ASVspoof2019CMDataset(ASV19_EVAL_AUDIO_DIR, ASV19_EVAL_PROTO)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=safe_collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=safe_collate,
    )
    eval19_loader = DataLoader(
        eval19_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=safe_collate,
    )

    # Sanity checks (anti-collapse)
    sanity_check(model, train_loader, device, name="(ASV19-train)")
    sanity_check(model, dev_loader, device, name="(ASV19-dev)")

    # Loss (collapse-proof): logits + BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_dev_eer = 1e9

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_eer = eval_eer(model, dev_loader, device, name="dev")

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | dev_EER={dev_eer:.4f}")

        # Collapse detector (quick probe)
        with torch.no_grad():
            probe = None
            for _ in range(20):
                b = next(iter(dev_loader))
                if b is not None:
                    probe = b
                    break
            if probe is not None:
                xb, _, _ = probe
                xb = xb.to(device)
                lg = model(xb)
                if lg.std().item() < 1e-4:
                    print("🚨 COLLAPSE WARNING: logits std extremely small (<1e-4).")

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Saved BEST model: {SAVE_PATH} (best_dev_EER={best_dev_eer:.4f})")

    # Load best model for evaluations
    print("\n--- Loading BEST model for evaluations ---")
    model.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
    model.to(device)

    # ASV19 eval
    print("\n--- Evaluation on ASVspoof2019 LA eval ---")
    eval19_eer = eval_eer(model, eval19_loader, device, name="ASV19-eval")
    print(f"✅ ASVspoof2019 LA eval EER: {eval19_eer:.4f}")
    save_scores(model, eval19_loader, device, ASV19_SCORE_PATH, name="ASV19-eval")

    # ASV21 eval (corrupted skip enabled)
    print("\n--- Cross-domain Evaluation on ASVspoof2021 LA eval ---")
    eval21_ds = ASVspoof2021LAEvalDataset(ASV21_LA_EVAL_ROOT)
    print("ASV21 audio dir:", eval21_ds.audio_dir)
    print("ASV21 protocol :", eval21_ds.proto_path if eval21_ds.proto_path else "(not found)")
    print("ASV21 has labels:", eval21_ds.has_labels)

    eer21 = eval_asv21_and_save(model, eval21_ds, device, ASV21_SCORE_PATH)
    if eer21 is not None:
        print(f"✅ ASVspoof2021 LA eval EER: {eer21:.4f}")
    else:
        print("ℹ️ ASVspoof2021 labels not found; scores saved only.")

    print("\n✅ DONE")
    print("Saved model:", SAVE_PATH)
    print("ASV19 scores:", ASV19_SCORE_PATH)
    print("ASV21 scores:", ASV21_SCORE_PATH)

if __name__ == "__main__":
    main()
