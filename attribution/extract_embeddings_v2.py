"""Extract exemplar + query embeddings for all tasks and generators.

Task/clip combos: musician_3, musician_6, instrument_3, instrument_6,
                  genre_3, genre_6, melody_1

Output structure:
  embeddings/<encoder>/<generator>/<task_clip>/<concept>/exemplars.npz
  embeddings/<encoder>/<generator>/<task_clip>/<concept>/queries.npz

Each npz:
  emb          (N, D)  float32, L2-normalized
  files        (N,)    str
  prompt_type  (N,)    str  {'exemplar','contextual','style'}

Usage:
  python extract_embeddings_v2.py --encoder mert-v0 --task musician_6 --generator audioldm2
  python extract_embeddings_v2.py --encoder mert-v0 --task all --generator all
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from encoders import build_encoder

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA     = Path("/path/to/truemuse-data/data")       # update to <dataset_root>/data
OUT_ROOT = Path("/path/to/truemuse-data/embeddings") # update to <dataset_root>/embeddings

# Exemplar roots per task (concept audio clips)
EXEMPLAR_ROOTS = {
    "musician_3":   DATA / "concepts" / "musician_3",
    "musician_6":   DATA / "concepts" / "musician_6",
    "instrument_3": DATA / "concepts" / "instrument_3",
    "instrument_6": DATA / "concepts" / "instrument_6",
    "genre_3":      DATA / "concepts" / "genre_3",
    "genre_6":      DATA / "concepts" / "genre_6",
    "melody_1":     DATA / "concepts" / "melody_1",
}

# Generated roots per (task, generator): (root_dir, concept_suffix)
GENERATED = {
    "musician_3": {
        "audioldm2":    (DATA / "generated" / "audioldm2"    / "musician_3",  "_ldm2_db"),
        "mustango":     (DATA / "generated" / "mustango"     / "musician_3",  "_mustango_db"),
        "stable-audio": (DATA / "generated" / "stable-audio" / "musician_3",  "_sa_db"),
    },
    "musician_6": {
        "audioldm2":    (DATA / "generated" / "audioldm2"    / "musician_6",  "_ldm2_db"),
        "mustango":     (DATA / "generated" / "mustango"     / "musician_6",  "_mustango_db"),
        "stable-audio": (DATA / "generated" / "stable-audio" / "musician_6",  "_sa_db"),
    },
    "instrument_3": {
        "audioldm2":    (DATA / "generated" / "audioldm2"    / "instrument_3", "_ldm2_db"),
        "mustango":     (DATA / "generated" / "mustango"     / "instrument_3", "_mustango_db"),
        "stable-audio": (DATA / "generated" / "stable-audio" / "instrument_3", "_sa_db"),
    },
    "instrument_6": {
        "audioldm2":    (DATA / "generated" / "audioldm2"    / "instrument_6", "_ldm2_db"),
        "mustango":     (DATA / "generated" / "mustango"     / "instrument_6", "_mustango_db"),
        "stable-audio": (DATA / "generated" / "stable-audio" / "instrument_6", "_sa_db"),
    },
    "genre_3": {
        "audioldm2":    (DATA / "generated" / "audioldm2"    / "genre_3",     "_ldm2_db"),
        "mustango":     (DATA / "generated" / "mustango"     / "genre_3",     "_mustango_db"),
        "stable-audio": (DATA / "generated" / "stable-audio" / "genre_3",     "_sa_db"),
    },
    "genre_6": {
        "audioldm2":    (DATA / "generated" / "audioldm2"    / "genre_6",     "_ldm2_db"),
        "mustango":     (DATA / "generated" / "mustango"     / "genre_6",     "_mustango_db"),
        "stable-audio": (DATA / "generated" / "stable-audio" / "genre_6",     "_sa_db"),
    },
    "melody_1": {
        "audioldm2":    (DATA / "generated" / "audioldm2"    / "melody_1",    "_ldm2_db"),
        "mustango":     (DATA / "generated" / "mustango"     / "melody_1",    "_mustango_db"),
        "stable-audio": (DATA / "generated" / "stable-audio" / "melody_1",    "_sa_db"),
    },
}

ALL_TASKS      = list(EXEMPLAR_ROOTS.keys())
ALL_GENERATORS = ["audioldm2", "mustango", "stable-audio"]
ALL_ENCODERS   = ["mert-v0", "mert-v0-public", "music2vec-v1", "dac-16k"]


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_wav_mono(path):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).numpy().astype(np.float32), sr


def exemplar_wavs(ex_dir):
    return sorted(glob.glob(str(ex_dir / "*.wav")))


def query_wavs(gen_dir):
    """Return [(path, prompt_type)] for contextual and style subdirs."""
    out = []
    for sub in ("contextual", "style"):
        for p in sorted(glob.glob(str(gen_dir / sub / "*.wav"))):
            out.append((p, sub))
    return out


def embed_files(encoder, path_type_pairs, rel_to):
    embs, files, types = [], [], []
    for p, pt in path_type_pairs:
        try:
            wav, sr = load_wav_mono(p)
            emb = encoder.embed(wav, sr)
        except Exception as e:
            print(f"    skip {p}: {e}", flush=True)
            continue
        embs.append(emb)
        files.append(os.path.relpath(p, str(rel_to)))
        types.append(pt)
    if not embs:
        return None
    return dict(
        emb=np.stack(embs).astype(np.float32),
        files=np.array(files),
        prompt_type=np.array(types),
    )


# ── Core processing ────────────────────────────────────────────────────────────

def process(encoder, encoder_name, generator, task, overwrite=False):
    gen_root, suffix = GENERATED[task][generator]
    ex_root = EXEMPLAR_ROOTS[task]

    if not gen_root.is_dir():
        print(f"  WARN generated root missing: {gen_root}", flush=True)
        return

    # concepts = dirs in gen_root that end with the generator suffix
    concepts = sorted(
        d.name[: -len(suffix)]
        for d in gen_root.iterdir()
        if d.is_dir() and d.name.endswith(suffix)
    )
    print(f"\n[{generator}/{task}] {len(concepts)} concepts", flush=True)

    out_dir = OUT_ROOT / encoder_name / generator / task
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, concept in enumerate(concepts):
        concept_out = out_dir / concept
        concept_out.mkdir(exist_ok=True)
        ex_npz = concept_out / "exemplars.npz"
        q_npz  = concept_out / "queries.npz"

        ex_dir  = ex_root / concept
        gen_dir = gen_root / (concept + suffix)
        print(f"  [{i+1}/{len(concepts)}] {concept}", flush=True)

        if not ex_dir.is_dir():
            print(f"    WARN no exemplar dir: {ex_dir}", flush=True)
            continue

        # exemplars
        if overwrite or not ex_npz.exists():
            wavs = exemplar_wavs(ex_dir)
            if wavs:
                data = embed_files(encoder, [(w, "exemplar") for w in wavs], DATA)
                if data:
                    np.savez(ex_npz, **data)
                    print(f"    exemplars: {len(wavs)} clips saved", flush=True)
            else:
                print(f"    WARN no exemplar wavs in {ex_dir}", flush=True)
        else:
            print(f"    exemplars cached", flush=True)

        # queries
        if overwrite or not q_npz.exists():
            pairs = query_wavs(gen_dir)
            if pairs:
                data = embed_files(encoder, pairs, DATA)
                if data:
                    np.savez(q_npz, **data)
                    print(f"    queries:   {len(pairs)} wavs saved", flush=True)
            else:
                print(f"    WARN no query wavs in {gen_dir}", flush=True)
        else:
            print(f"    queries cached", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder",   required=True,
                    choices=ALL_ENCODERS + ["all"])
    ap.add_argument("--generator", required=True,
                    choices=ALL_GENERATORS + ["all"])
    ap.add_argument("--task",      required=True,
                    choices=ALL_TASKS + ["all"])
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    encoders   = ALL_ENCODERS   if args.encoder   == "all" else [args.encoder]
    generators = ALL_GENERATORS if args.generator == "all" else [args.generator]
    tasks      = ALL_TASKS      if args.task      == "all" else [args.task]

    for enc_name in encoders:
        print(f"\nLoading encoder: {enc_name}", flush=True)
        encoder = build_encoder(enc_name, device=args.device)
        for gen in generators:
            for task in tasks:
                process(encoder, enc_name, gen, task, overwrite=args.overwrite)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
