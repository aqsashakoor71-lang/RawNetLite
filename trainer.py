import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

from focal_loss import FocalLoss

# ✅ Multi-head model
from RawNetLite import RawNetLite

# ✅ This dataset uses .pt tensors (preprocessed)
from FOR_dataset import FakeOrRealTestDataset

# ❌ These are NOT used in single-domain training; keeping as comments (unwanted)
# from AVSpoof_dataset import AVSpoofTestDataset
# from CodecFake_dataset import CodecFakeTestDataset
from torch.utils.data import random_split, DataLoader

# ❌ Not using mixed datasets in this experiment; keep commented
# from Mixed_dataset import DoubleDomainDataset, MultiDomainDataset, AugmentedMultiDomainDataset


# ------------------------------
# PARAMETERS
# ------------------------------
BATCH_SIZE = 16          # Batch size
EPOCHS = 20              # You can set 1 for quick demo
LEARNING_RATE = 1e-4
SEED = 42

# Max samples
MAX_REAL = 5000
MAX_FAKE = 5000

LOSS = "focal"           # "focal" or "bce"

# ------------------------------
# DATASET CONFIGURATION
# ------------------------------
# We want ONLY single-domain (ASVspoof2019-LA processed tensors)
CROSS_DOMAIN = False
TRIPLE_DOMAIN = False
AUGMENTATION = False

# ------------------------------
# PATHS
# ------------------------------
MODEL_ROOT = os.path.join(os.getcwd(), "models")
MODEL_NAME = "rawnetlite_multitask_cm_quality.pt"

# ✅ This MUST exist and contain real_processed/ fake_processed
DATASET_ROOT_FOR = "/kaggle/working/asv19_la_train_processed"

# ❌ not used but kept
DATASET_ROOT_AVSPOOF = "/kaggle/working/dummy_avspoof"
DATASET_ROOT_CODECFAKE = "/kaggle/working/dummy_codecfake"


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


def print_simple_classification_report(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)

    for cls in [0, 1]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        support = np.sum(y_true == cls)

        print(f"class {cls}  prec: {precision:0.4f}  rec: {recall:0.4f}  f1: {f1:0.4f}  support: {support}")


# ------------------------------
# QUALITY (CLEAN vs DEGRADED)
# ------------------------------
def degrade_waveform(waveforms):
    """
    cheap codec-ish degradation
    waveforms: [B, 1, T]
    """
    x = F.avg_pool1d(waveforms, kernel_size=4, stride=4)
    x = x.repeat_interleave(4, dim=-1)
    if x.size(-1) > waveforms.size(-1):
        x = x[..., :waveforms.size(-1)]
    elif x.size(-1) < waveforms.size(-1):
        x = F.pad(x, (0, waveforms.size(-1) - x.size(-1)))
    return x


def make_quality_labels(batch_size, device, p_degrade=0.5):
    """
    0 = clean, 1 = degraded
    """
    return (torch.rand(batch_size, device=device) < p_degrade).long()


# ------------------------------
# DATASET LOADING
# ------------------------------
def load_dataset():
    """
    We use FakeOrRealTestDataset which expects:
    /output_root/real_processed/*.pt
    /output_root/fake_processed/*.pt
    """
    if not CROSS_DOMAIN:
        print("[INFO] Using FakeOrRealTestDataset (single-domain, ASVspoof2019-LA)")
        dataset = FakeOrRealTestDataset(
            real_dir=os.path.join(DATASET_ROOT_FOR, "real_processed"),
            fake_dir=os.path.join(DATASET_ROOT_FOR, "fake_processed"),
            max_real=MAX_REAL,
            max_fake=MAX_FAKE,
        )
        return dataset

    # ❌ not used
    raise ValueError("CROSS_DOMAIN=True is not supported in this file (kept commented).")


# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(MODEL_ROOT, exist_ok=True)

    dataset = load_dataset()
    dataset_size = len(dataset)
    print(f"[INFO] Total samples in dataset: {dataset_size}")

    if dataset_size == 0:
        raise ValueError("Dataset is empty. Check preprocessing output_root folders.")

    # Split 80/10/10
    train_len = int(0.8 * dataset_size)
    val_len = int(0.1 * dataset_size)
    test_len = dataset_size - train_len - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    # num_workers=0 for Kaggle stability
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = RawNetLite().to(device)

    # CM loss
    if LOSS == "focal":
        criterion_cm = FocalLoss(alpha=0.25, gamma=2.0)
    elif LOSS == "bce":
        criterion_cm = nn.BCELoss()
    else:
        raise ValueError("Invalid loss function. Choose 'focal' or 'bce'.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0

    # --------------------------
    # EPOCH LOOP
    # --------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            waveforms = waveforms.to(device).float()   # [B,1,T]
            labels = labels.to(device).float()         # [B]

            # ---- QUALITY LABELS + DEGRADE HALF ----
            q_labels = make_quality_labels(waveforms.size(0), device=device)  # [B]
            waveforms_degraded = degrade_waveform(waveforms)
            mask = (q_labels == 1).view(-1, 1, 1)
            waveforms_used = torch.where(mask, waveforms_degraded, waveforms)

            # ---- MULTI-HEAD FORWARD ----
            cm_logits, q_logits = model(waveforms_used)            # cm:[B,1], q:[B,2]
            cm_prob = torch.sigmoid(cm_logits).squeeze()           # [B]

            # ---- LOSSES ----
            loss_cm = criterion_cm(cm_prob, labels)
            loss_q = F.cross_entropy(q_logits, q_labels)
            loss = loss_cm + 0.3 * loss_q

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

        # ---------- VALIDATION (CM only) ----------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                waveforms = waveforms.to(device).float()
                labels = labels.to(device).float()

                cm_logits, q_logits = model(waveforms)
                cm_prob = torch.sigmoid(cm_logits).squeeze()
                preds = (cm_prob > 0.5).float()

                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        acc = simple_accuracy(y_true, y_pred)
        f1 = simple_f1(y_true, y_pred)
        print(f"Validation Accuracy: {acc:.4f} - F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(MODEL_ROOT, MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model at epoch {epoch+1} with F1 = {f1:.4f}")

    # --------------------------
    # TEST PHASE (best model)
    # --------------------------
    print("\n[INFO] Evaluation on test set with best saved model:")
    best_model_path = os.path.join(MODEL_ROOT, MODEL_NAME)
    model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Testing"):
            waveforms = waveforms.to(device).float()
            labels = labels.to(device).float()

            cm_logits, q_logits = model(waveforms)
            cm_prob = torch.sigmoid(cm_logits).squeeze()
            preds = (cm_prob > 0.5).float()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = simple_accuracy(y_true, y_pred)
    f1 = simple_f1(y_true, y_pred)
    print("\n[TEST RESULTS] on ASVspoof2019-LA (single-domain)")
    print(f"Test Accuracy: {acc:.4f} - Test F1: {f1:.4f}")
    print("\nSimple classification report:")
    print_simple_classification_report(y_true, y_pred)


if __name__ == "__main__":
    train()
