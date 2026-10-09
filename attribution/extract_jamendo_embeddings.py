"""Extract embeddings for Jamendo distractor pool.

Outputs:
  embeddings/jamendo/<encoder>/pool.npz
    emb   (N, D)  float32, L2-normalized
    files (N,)    str, filename

Usage:
  python extract_jamendo_embeddings.py --encoder mert-v0
  python extract_jamendo_embeddings.py --encoder all
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

JAMENDO_DIR = Path("/path/to/Data-attribution/AudioAttribution/Jamendo")
OUT_ROOT    = Path("/path/to/Data-attribution/AudioAttribution/embeddings/jamendo")
ENCODERS    = ["mert-v0", "mert-v0-public", "music2vec-v1", "dac-16k"]

sys.path.insert(0, str(Path(__file__).parent))
from encoders import build_encoder


CLIP_SEC    = 10
SAVE_EVERY  = 500
BATCH_SIZE  = 16

TARGET_SR = 16000

def iter_chunks(path: Path):
    """Yield (wav, sr, offset) for every non-overlapping 10s chunk, pre-resampled to TARGET_SR."""
    import torchaudio
    wav_full, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != TARGET_SR:
        import torch
        wav_full = torchaudio.functional.resample(
            torch.from_numpy(wav_full), sr, TARGET_SR
        ).numpy()
    chunk_samples = int(CLIP_SEC * TARGET_SR)
    n_chunks = len(wav_full) // chunk_samples
    for i in range(n_chunks):
        chunk = wav_full[i * chunk_samples:(i + 1) * chunk_samples]
        yield chunk.astype(np.float32), TARGET_SR, float(i * CLIP_SEC)


def process_encoder(encoder_name: str, device: str, overwrite: bool):
    out_dir = OUT_ROOT / encoder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pool.npz"

    if out_path.exists() and not overwrite:
        print(f"[{encoder_name}] pool.npz already exists, skipping.")
        return

    mp3_files = sorted(JAMENDO_DIR.glob("*.mp3"))
    print(f"[{encoder_name}] Loading encoder...", flush=True)
    encoder = build_encoder(encoder_name, device=device)

    embs, files = [], []
    buf_wavs, buf_srs, buf_names = [], [], []

    def flush_batch():
        if not buf_wavs:
            return
        batch_embs = encoder.embed_batch(buf_wavs, buf_srs[0])
        embs.extend(batch_embs)
        files.extend(buf_names)
        buf_wavs.clear(); buf_srs.clear(); buf_names.clear()

    def save_checkpoint():
        if not embs:
            return
        np.savez(out_path,
                 emb=np.stack(embs).astype(np.float32),
                 files=np.array(files))
        print(f"  checkpoint: {len(embs)} embeddings -> {out_path}", flush=True)

    for i, mp3 in enumerate(mp3_files):
        try:
            for wav, sr, offset in iter_chunks(mp3):
                if buf_srs and sr != buf_srs[0]:
                    flush_batch()
                buf_wavs.append(wav)
                buf_srs.append(sr)
                buf_names.append(f"{mp3.name}@{offset:.0f}s")
                if len(buf_wavs) >= BATCH_SIZE:
                    flush_batch()
                    if len(embs) % SAVE_EVERY == 0:
                        save_checkpoint()
        except Exception as e:
            print(f"  skip {mp3.name}: {e}", flush=True)
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(mp3_files):
            print(f"  {i+1}/{len(mp3_files)} songs | {len(embs)} chunks", flush=True)

    flush_batch()
    save_checkpoint()
    if embs:
        print(f"[{encoder_name}] Done: {len(embs)} embeddings -> {out_path}")
    else:
        print(f"[{encoder_name}] No embeddings extracted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=ENCODERS + ["all"])
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    encoders = ENCODERS if args.encoder == "all" else [args.encoder]
    for enc in encoders:
        process_encoder(enc, args.device, args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()
