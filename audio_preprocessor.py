import torch
import argparse
import torchaudio
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def preprocess_audio(waveform, sr, target_sr: int = 16000, target_sec: float = 3.0):
    """
    Preprocess an input waveform for deep learning models.

    Steps:
    1. Stereo -> mono (average channels)
    2. Resample to target_sr
    3. Normalize to [-1, 1]
    4. Trim or pad to fixed duration target_sec
    """
    # 1) Stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 2) Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # 3) Normalize
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    # 4) Fix length
    num_samples = int(target_sr * target_sec)
    if waveform.shape[1] < num_samples:
        pad = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :num_samples]

    return waveform


def preprocess_from_csv(
    csv_path,
    input_dir,
    output_root,
    label_map=None,
    use_full_path: bool = False,
    class_limit: int | None = None,
):
    """
    Preprocess audio files listed in a CSV file and save them as .pt tensors.

    Expected CSV formats:

    1) RawNetLite-style (FakeOrReal / AVSpoof2021 / CodecFake):
       - columns like: path,label  (or filepath,label)
       - label: 'bonafide'/'spoof' or similar strings

    2) Our ASVspoof2019 metadata:
       - columns: path,label
       - path: full absolute path to .flac (when use_full_path=True)
       - label: 0 (bonafide), 1 (spoof)

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file.
    input_dir : str or Path
        Base directory for audio files (used if use_full_path=False).
    output_root : str or Path
        Directory where 'real_processed' and 'fake_processed' will be created.
    label_map : dict or None
        Optional mapping for string labels -> int.
    use_full_path : bool
        If True, CSV 'path' column is treated as full path; input_dir is ignored.
    class_limit : int or None
        Optional limit of samples per class (0 = bonafide, 1 = spoof).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    input_dir = Path(input_dir)
    output_root = Path(output_root)

    real_dir = output_root / "real_processed"
    fake_dir = output_root / "fake_processed"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Small backward-compatible note:
    # Old implementation used string labels and a label_map.
    # We keep that option but also support numeric labels (0/1) as in ASVspoof2019.
    real = 0
    fake = 0

    print(f"[INFO] Reading metadata from: {csv_path}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] use_full_path = {use_full_path}")
    print(f"[INFO] class_limit   = {class_limit}")

    # Try to find the 'path' column name (some CSVs might use 'filepath')
    if "path" in df.columns:
        path_col = "path"
    elif "filepath" in df.columns:
        path_col = "filepath"
    else:
        raise ValueError("CSV must contain a 'path' or 'filepath' column.")

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # --------- Resolve audio path ----------
        if use_full_path:
            audio_path = Path(row[path_col])
        else:
            audio_path = input_dir / str(row[path_col])

        # --------- Resolve label ----------
        label_raw = row["label"]

        if label_map is not None:
            # String-based mapping (e.g., 'bonafide' -> 0, 'spoof' -> 1)
            key = str(label_raw).lower()
            label = int(label_map[key])
        else:
            # Support both string and numeric labels directly
            if isinstance(label_raw, str):
                key = label_raw.lower()
                if key in ["bonafide", "bona-fide", "bona_fide", "real"]:
                    label = 0
                else:
                    label = 1
            else:
                label = int(label_raw)

        # --------- Optional class_limit ----------
        if class_limit is not None:
            if label == 0 and real >= class_limit:
                continue
            if label == 1 and fake >= class_limit:
                continue

        if not audio_path.exists():
            print(f"[WARN] Missing file: {audio_path}")
            continue

        save_dir = real_dir if label == 0 else fake_dir
        save_path = save_dir / (audio_path.stem + ".pt")

        if save_path.exists():
            # already processed
            continue

        try:
            wav, sr = torchaudio.load(audio_path)
            wav = preprocess_audio(wav, sr)
            torch.save(wav, save_path)
            if label == 0:
                real += 1
            else:
                fake += 1
        except Exception as e:
            print(f"[ERROR] Problem with {audio_path}: {e}")

    print(f"[DONE] Saved {real} real samples and {fake} fake samples at {output_root}")


# ======================================================================
# NOTE:
# Purani implementation yahan thi (argparse with positional arguments and
# direct label_map-only logic). Supervisor ke liye reference ke taur pe
# aap yahan commented block rakh sakte ho agar chahein, lekin main yahan
# sirf NEW, clean CLI rakha hoon.
# ======================================================================

if __name__ == "__main__":
    # Command-line interface used in README:
    #
    # python audio_preprocessor.py \
    #   --csv_path metadata.csv \
    #   --input_dir path/to/audio \
    #   --output_root data/audio_processed/
    #
    parser = argparse.ArgumentParser(
        description="Preprocess audio files listed in a CSV and save them as .pt tensors."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the input CSV file (with columns: path, label).",
        required=True,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Base directory of audio files (ignored if --use_full_path is set).",
        required=True,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Root directory where processed tensors will be stored.",
        required=True,
    )
    parser.add_argument(
        "--label_map",
        type=str,
        help="Optional path to a JSON file mapping string labels to integers.",
        required=False,
    )
    parser.add_argument(
        "--use_full_path",
        action="store_true",
        help="If set, CSV 'path' column is treated as full absolute path.",
    )
    parser.add_argument(
        "--class_limit",
        type=int,
        help="Optional limit of samples per class (0=real, 1=fake).",
        required=False,
    )

    args = parser.parse_args()

    # Optional label_map loading
    lm = None
    if args.label_map:
        import json

        with open(args.label_map, "r") as f:
            lm = json.load(f)

    preprocess_from_csv(
        csv_path=args.csv_path,
        input_dir=args.input_dir,
        output_root=args.output_root,
        label_map=lm,
        use_full_path=args.use_full_path,
        class_limit=args.class_limit,
    )
