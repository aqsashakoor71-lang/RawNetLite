import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model
from sklearn.metrics import accuracy_score, f1_score

from RawNetLite import RawNetLite

# ------------------------

# ------------------------
ASV19_ROOT = "/kaggle/input/asvpoof-2019-dataset/LA/LA"
TRAIN_AUDIO = os.path.join(ASV19_ROOT, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO = os.path.join(ASV19_ROOT, "ASVspoof2019_LA_dev", "flac")
PROTO_DIR = os.path.join(ASV19_ROOT, "ASVspoof2019_LA_cm_protocols")

TRAIN_PROTO = os.path.join(PROTO_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTO = os.path.join(PROTO_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

SAVE_PATH = "/kaggle/working/models/RawNetLite_SSL_Distill.pt"

SR = 16000
CLIP_LEN = 48000
BATCH = 16
EPOCHS = 20
LR = 1e-4
ALPHA = 0.5   # distillation weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("/kaggle/working/models", exist_ok=True)

# ------------------------
# DATA
# ------------------------
def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    # normalize
    m = np.max(np.abs(wav))
    if m > 0:
        wav = wav / m

    # pad/trim
    if len(wav) >= CLIP_LEN:
        wav = wav[:CLIP_LEN]
    else:
        wav = np.pad(wav, (0, CLIP_LEN - len(wav)))

    return wav

def parse_proto(proto):
    items = []
    with open(proto) as f:
        for line in f:
            p = line.strip().split()
            fid = p[1]
            y = 0 if p[-1] == "bonafide" else 1
            items.append((fid, y))
    return items

class ASV19Dataset(Dataset):
    def __init__(self, audio_dir, proto):
        self.items = []
        for fid, y in parse_proto(proto):
            path = os.path.join(audio_dir, fid + ".flac")
            if os.path.exists(path):
                self.items.append((path, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        wav = load_audio(path)
        return torch.tensor(wav).unsqueeze(0), y

# ------------------------
# LOAD TEACHER
# ------------------------
teacher = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False

# ------------------------
# STUDENT
# ------------------------
student = RawNetLite().to(device)

optimizer = torch.optim.Adam(student.parameters(), lr=LR)
bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()

# ------------------------
# LOADERS
# ------------------------
train_loader = DataLoader(
    ASV19Dataset(TRAIN_AUDIO, TRAIN_PROTO),
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

dev_loader = DataLoader(
    ASV19Dataset(DEV_AUDIO, DEV_PROTO),
    batch_size=BATCH,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ------------------------
# TRAIN
# ------------------------
for epoch in range(EPOCHS):
    student.train()
    total_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x = x.to(device)
        y = y.float().to(device).unsqueeze(1)

        # Teacher embedding
        with torch.no_grad():
            teacher_out = teacher(x.squeeze(1)).last_hidden_state
            teacher_embed = teacher_out.mean(dim=1)

        logits, student_embed = student(x)

        loss_cls = bce(logits, y)
        loss_kd = mse(student_embed, teacher_embed)

        loss = (1-ALPHA)*loss_cls + ALPHA*loss_kd

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(student.state_dict(), SAVE_PATH)
print("✅ Saved:", SAVE_PATH)
