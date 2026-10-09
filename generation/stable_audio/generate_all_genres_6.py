"""
Batch generate audio for all genre_models_6 (6-clip, 1500-step, Stable Audio).
Run with: nohup python generate_all_genres_6.py > generate_all_genres_6.log 2>&1 &
"""

import gc
import os
import sys
import csv
import torch
from einops import rearrange
from scipy.io.wavfile import write

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# ── Config ────────────────────────────────────────────────────────────────────

BASE      = "/path/to/Data-attribution/stable-audio-tools"
MODEL_DIR = os.path.join(BASE, "genre_models_6")
OUT_BASE  = os.path.join(BASE, "generated_samples_genres_6")

CKPT_STEP      = 1500
INSTANCE_WORD  = "sks"
OBJECT_CLASS   = "genre"
NUM_STEPS      = 200
CFG_SCALE      = 4.0
NUM_WAVEFORMS  = 4
WINDOW_SECONDS = 10
SAMPLER_TYPE   = "dpmpp-3m-sde"
SIGMA_MIN      = 0.3
SIGMA_MAX      = 500.0


def list_genres():
    out = []
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.endswith("_sa_db"):
            continue
        ckpt = os.path.join(MODEL_DIR, name, f"model_step_{CKPT_STEP}", "pytorch_model.ckpt")
        if os.path.exists(ckpt):
            out.append(name)
    return out


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    return [
        f"Driving instrumental, electric guitar riff, 120 BPM, in {instance_word} {object_class}",
        f"Heavy anthem, distorted power chords, 130 BPM, in {instance_word} {object_class}",
        f"Atmospheric soundscape, synthesizer pads, 110 BPM, in {instance_word} {object_class}",
        f"Energetic dance track, pulsing beat, 128 BPM, in {instance_word} {object_class}",
        f"Upbeat melodic tune, catchy hook, 115 BPM, in {instance_word} {object_class}",
        f"Bright ballad, piano and vocals, 95 BPM, in {instance_word} {object_class}",
        f"Experimental ambient soundscape, textured drone, 90 BPM, in {instance_word} {object_class}",
        f"Hard hitting beat, deep bass, 90 BPM, in {instance_word} {object_class}",
        f"Laid back lo-fi loop, vinyl crackle, 85 BPM, in {instance_word} {object_class}",
        f"Orchestral piece, strings and woodwinds, 80 BPM, in {instance_word} {object_class}",
        f"Dramatic crescendo, full orchestra, 100 BPM, in {instance_word} {object_class}",
        f"Warm acoustic melody, fingerpicked guitar, 100 BPM, in {instance_word} {object_class}",
        f"Traditional tune, acoustic ensemble, 105 BPM, in {instance_word} {object_class}",
        f"Smooth groove, walking bass line, 110 BPM, in {instance_word} {object_class}",
        f"Acoustic ballad, steel guitar, 95 BPM, in {instance_word} {object_class}",
    ]


def get_style_prompts(instance_word, object_class):
    instruments = [
        ("piano", "90 BPM"), ("acoustic guitar", "105 BPM"), ("electric guitar", "120 BPM"),
        ("violin", "85 BPM"), ("saxophone", "100 BPM"), ("trumpet", "110 BPM"),
        ("drums", "128 BPM"), ("bass guitar", "115 BPM"), ("synthesizer", "124 BPM"),
        ("flute", "95 BPM"), ("cello", "80 BPM"), ("harp", "70 BPM"),
        ("organ", "90 BPM"), ("clarinet", "100 BPM"), ("harmonica", "105 BPM"),
        ("accordion", "110 BPM"), ("banjo", "120 BPM"), ("ukulele", "100 BPM"),
        ("marimba", "95 BPM"), ("vibraphone", "90 BPM"), ("tabla", "110 BPM"),
        ("djembe", "115 BPM"), ("sitar", "85 BPM"), ("erhu", "80 BPM"),
        ("kalimba", "90 BPM"), ("steel drums", "105 BPM"), ("bagpipes", "100 BPM"),
        ("theremin", "75 BPM"),
    ]
    return [f"Energetic {inst} performance, rich arrangement, {bpm}, in {instance_word} {object_class}"
            for inst, bpm in instruments]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_config


# ── Inference ─────────────────────────────────────────────────────────────────

def save_audio(audio, out_path, sample_rate):
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.float().div(torch.max(torch.abs(audio))).clamp(-1, 1)
    audio_np = audio.cpu().numpy().T
    write(out_path, sample_rate, audio_np)


@torch.no_grad()
def generate(model, model_config, prompt, out_path_prefix, device):
    sample_rate = model_config["sample_rate"]
    sample_size = sample_rate * WINDOW_SECONDS

    existing = [f"{out_path_prefix}_s{j+1}.wav" for j in range(NUM_WAVEFORMS)]
    if all(os.path.exists(p) for p in existing):
        print(f"  Skipped (already exists): {out_path_prefix}", flush=True)
        return existing

    paths = []
    for j in range(NUM_WAVEFORMS):
        audio = generate_diffusion_cond(
            model,
            steps=NUM_STEPS, cfg_scale=CFG_SCALE,
            conditioning=[{"prompt": prompt, "seconds_start": 0, "seconds_total": WINDOW_SECONDS}],
            sample_size=sample_size, device=device,
            sampler_type=SAMPLER_TYPE, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
        )
        out_path = f"{out_path_prefix}_s{j+1}.wav"
        save_audio(audio, out_path, sample_rate)
        print(f"  Saved: {out_path}", flush=True)
        paths.append(out_path)

    gc.collect()
    torch.cuda.empty_cache()
    return paths


def process_genre(genre, device):
    ckpt_path = os.path.join(MODEL_DIR, genre, f"model_step_{CKPT_STEP}", "pytorch_model.ckpt")
    out_dir   = os.path.join(OUT_BASE, genre)

    print(f"\nLoading model: {ckpt_path}", flush=True)
    model, model_config = load_model(ckpt_path, device)
    records = []

    ctx_dir = os.path.join(out_dir, "contextual")
    os.makedirs(ctx_dir, exist_ok=True)
    print("[Contextual prompts]", flush=True)
    for i, prompt in enumerate(get_contextual_prompts(INSTANCE_WORD, OBJECT_CLASS)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"ctx_{i+1:02d}"
        paths = generate(model, model_config, prompt, os.path.join(ctx_dir, slug), device)
        for p in paths:
            records.append({"genre": genre, "step": CKPT_STEP,
                             "prompt_type": "contextual", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    style_dir = os.path.join(out_dir, "style")
    os.makedirs(style_dir, exist_ok=True)
    print("[Style prompts]", flush=True)
    for i, prompt in enumerate(get_style_prompts(INSTANCE_WORD, OBJECT_CLASS)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"style_{i+1:02d}"
        paths = generate(model, model_config, prompt, os.path.join(style_dir, slug), device)
        for p in paths:
            records.append({"genre": genre, "step": CKPT_STEP,
                             "prompt_type": "style", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "step", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    genres = list_genres()
    print(f"Found {len(genres)} genres with model_step_{CKPT_STEP}", flush=True)

    all_records = []
    for genre in genres:
        print(f"\n{'='*50}", flush=True)
        print(f"Genre: {genre}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_genre(genre, device))
        except Exception as e:
            import traceback
            print(f"ERROR on {genre}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all_sa.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "step", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
