import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from focal_loss import FocalLoss
from RawNetLite import RawNetLite
from FOR_dataset import FakeOrRealTestDataset
from AVSpoof_dataset import AVSpoofTestDataset
from CodecFake_dataset import CodecFakeTestDataset
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from Mixed_dataset import DoubleDomainDataset, MultiDomainDataset, AugmentedMultiDomainDataset

# ------------------------------
# PARAMETERS
# ------------------------------
BATCH_SIZE = 16          # Batch size for DataLoaders
EPOCHS = 35              # Number of epochs for training
LEARNING_RATE = 1e-4     # Learning rate for the optimizer
SEED = 42                # Random seed for reproducibility

# Max samples for each dataset
MAX_PER_CLASS = 5000     # Max samples per class (for cross-domain / multi-dataset)
MAX_REAL = 5000          # Max real samples (single-domain)
MAX_FAKE = 5000          # Max fake samples (single-domain)

LOSS = "focal"           # "focal" or "bce"

# ------------------------------
# DATASET CONFIGURATION
# ------------------------------

# ORIGINAL (kept for reference, now commented):
# CROSS_DOMAIN = True   # If False, single domain mode is used
# TRIPLE_DOMAIN = True  # If True and CROSS_DOMAIN is True, triple domain mode is used
# AUGMENTATION = True   # If True and CROSS_DOMAIN and TRIPLE_DOMAIN is True, augmented triple domain mode is used

# UPDATED for our experiment:
# We want to train ONLY on ASVspoof2019-LA (single-domain training)
CROSS_DOMAIN = False      # -> FakeOrRealTestDataset only
TRIPLE_DOMAIN = False
AUGMENTATION = False

# ------------------------------
# FOLDERS
# ------------------------------

# ORIGINAL PLACEHOLDERS (kept as comments):
# MODEL_ROOT = os.path.join(os.getcwd(), "path/to", "models")
# MODEL_NAME = "model_name_to_be_used.pt"
# DATASET_ROOT_FOR = os.path.join(os.getcwd(), "path/to", "FOR")
# DATASET_ROOT_AVSPOOF = os.path.join(os.getcwd(), "path/to", "AVSpoof2021")
# DATASET_ROOT_CODECFAKE = os.path.join(os.getcwd(), "path/to", "CodecFake")

# UPDATED FOR KAGGLE + ASVSPOOF2019
# 1) Where to save the trained model
MODEL_ROOT = os.path.join(os.getcwd(), "models")
MODEL_NAME = "rawnetlite_asv19.pt"

# 2) ASVspoof2019-LA processed tensors (already created by audio_preprocessor.py)
#    /kaggle/working/asv19_la_train_processed/real_processed
#    /kaggle/working/asv19_la_train_processed/fake_processed
DATASET_ROOT_FOR = "/kaggle/working/asv19_la_train_processed"

# 3) These roots are NOT used when CROSS_DOMAIN = False,
#    but we keep them to satisfy the original interface.
DATASET_ROOT_AVSPOOF = "/kaggle/working/dummy_avspoof"
DATASET_ROOT_CODECFAKE = "/kaggle/working/dummy_codecfake"


# ------------------------------
# DATASET LOADING
# ------------------------------
def load_dataset():
    """
    Build a dataset object depending on configuration flags.

    - If CROSS_DOMAIN == False:
        Single-domain RawNetLite using FakeOrRealTestDataset
        (here we point it to ASVspoof2019-LA processed tensors).

    - If CROSS_DOMAIN == True and TRIPLE_DOMAIN == False:
        DoubleDomainDataset using FOR + AVSpoof2021.

    - If CROSS_DOMAIN == True and TRIPLE_DOMAIN == True and AUGMENTATION == False:
        MultiDomainDataset using FOR + AVSpoof2021 + CodecFake.

    - If CROSS_DOMAIN == True and TRIPLE_DOMAIN == True and AUGMENTATION == True:
        AugmentedMultiDomainDataset with additional augmentation.
    """

    if not CROSS_DOMAIN:
        # Single-domain RawNetLite (our case: ASVspoof2019-LA)
        print("[INFO] Using FakeOrRealTestDataset (single-domain, ASVspoof2019-LA)")
        dataset = FakeOrRealTestDataset(
            real_dir=os.path.join(DATASET_ROOT_FOR, "real_processed"),
            fake_dir=os.path.join(DATASET_ROOT_FOR, "fake_processed"),
            max_real=MAX_REAL,
            max_fake=MAX_FAKE,
        )

    else:
        # Cross-domain settings: FOR + AVSpoof2021 (+ CodecFake)
        real_dirs = [
            os.path.join(DATASET_ROOT_FOR, "real_processed"),
            os.path.join(DATASET_ROOT_AVSPOOF, "real_processed"),
        ]
        fake_dirs = [
            os.path.join(DATASET_ROOT_FOR, "fake_processed"),
            os.path.join(DATASET_ROOT_AVSPOOF, "fake_processed"),
        ]

        if CROSS_DOMAIN and not TRIPLE_DOMAIN:
            # Double domain RawNetLite (FOR + AVSpoof2021)
            print("[INFO] Using DoubleDomainDataset (FOR + AVSpoof2021)")
            dataset = DoubleDomainDataset(
                real_dirs=real_dirs,
                fake_dirs=fake_dirs,
                max_per_class=MAX_PER_CLASS,
            )

        elif CROSS_DOMAIN and TRIPLE_DOMAIN and not AUGMENTATION:
            # Triple domain RawNetLite (FOR + AVSpoof2021 + CodecFake)
            print("[INFO] Using MultiDomainDataset (triple-domain)")
            real_dirs.append(os.path.join(DATASET_ROOT_CODECFAKE, "real_processed"))
            fake_dirs.append(os.path.join(DATASET_ROOT_CODECFAKE, "fake_processed"))
            dataset = MultiDomainDataset(
                real_dirs=real_dirs,
                fake_dirs=fake_dirs,
                max_per_class=MAX_PER_CLASS,
            )

        elif CROSS_DOMAIN and TRIPLE_DOMAIN and AUGMENTATION:
            # Augmented triple-domain RawNetLite
            print("[INFO] Using AugmentedMultiDomainDataset (augmented triple-domain)")
            real_dirs.append(os.path.join(DATASET_ROOT_CODECFAKE, "real_processed"))
            fake_dirs.append(os.path.join(DATASET_ROOT_CODECFAKE, "fake_processed"))
            dataset = AugmentedMultiDomainDataset(
                real_dirs=real_dirs,
                fake_dirs=fake_dirs,
                max_per_class=MAX_PER_CLASS,
            )

        else:
            raise ValueError("Invalid dataset configuration. Please check the parameters.")

    return dataset


# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train():
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Make sure model directory exists
    os.makedirs(MODEL_ROOT, exist_ok=True)

    # Load dataset according to configuration
    dataset = load_dataset()
    dataset_size = len(dataset)
    print(f"[INFO] Total samples in dataset: {dataset_size}")

    # Train/validation/test split (80/10/10)
    train_len = int(0.8 * dataset_size)
    val_len = int(0.1 * dataset_size)
    test_len = dataset_size - train_len - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = RawNetLite().to(device)

    # Loss function
    if LOSS == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif LOSS == "bce":
        criterion = nn.BCELoss()
    else:
        raise ValueError("Invalid loss function. Choose 'focal' or 'bce'.")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0

    # --------------------------
    # EPOCH LOOP
    # --------------------------
    for epoch in range(EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0

        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            waveforms = waveforms.to(device).float()
            labels = labels.to(device).float()

            outputs = model(waveforms).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                waveforms = waveforms.to(device).float()
                labels = labels.to(device).float()

                outputs = model(waveforms).squeeze()
                preds = (outputs > 0.5).float()

                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Validation Accuracy: {acc:.4f} - F1 Score: {f1:.4f}")

        # Save best model based on F1
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(MODEL_ROOT, MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model at epoch {epoch+1} with F1 = {f1:.4f}")

    # --------------------------
    # TEST PHASE (using best model)
    # --------------------------
    print("\n[INFO] Evaluation on test set with best saved model:")
    best_model_path = os.path.join(MODEL_ROOT, MODEL_NAME)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="Testing"):
            waveforms = waveforms.to(device).float()
            labels = labels.to(device).float()

            outputs = model(waveforms).squeeze()
            preds = (outputs > 0.5).float()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("\n[TEST RESULTS] on ASVspoof2019-LA (single-domain)")
    print(f"Test Accuracy: {acc:.4f} - Test F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    train()
