"""
Batch generate audio for all genre_models_6 (6-clip, 1500-step, Mustango).
Run with: nohup python generate_all_genres_6.py > generate_all_genres_6.log 2>&1 &
"""

import gc
import os
import sys
import csv
import json
import torch
from diffusers import DDPMScheduler
from scipy.io.wavfile import write

from models import MusicAudioDiffusion
from mustango import MusicFeaturePredictor
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL

# ── Config ────────────────────────────────────────────────────────────────────

PRETRAINED_PATH = "/path/to/mustango-pretrained"
BASE      = "/path/to/Data-attribution/mustango"
MODEL_DIR = os.path.join(BASE, "genre_models_6")
OUT_BASE  = os.path.join(BASE, "generated_samples_genres_6")

MODEL_STEP     = 1500
NUM_STEPS      = 100
GUIDANCE_SCALE = 3.0
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000


def list_genres():
    out = []
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.endswith("_mustango_db"):
            continue
        if os.path.isdir(os.path.join(MODEL_DIR, name, f"model_step_{MODEL_STEP}")):
            out.append(name)
    return out


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    return [
        f"This is a live concert recording in {instance_word} {object_class}, with a cheering crowd in the background. The atmosphere is electrifying.",
        f"This is a late-night studio session in {instance_word} {object_class}, laying down a slow groove over warm Rhodes keys. The atmosphere is intimate.",
        f"This is a rooftop sunset set in {instance_word} {object_class}, a smooth downtempo track with soft percussion. The atmosphere is mellow.",
        f"This is a rainy-day session in {instance_word} {object_class}, a reflective piece with soft piano and distant thunder. The atmosphere is melancholic.",
        f"This is a morning cafe session in {instance_word} {object_class}, a light tune with acoustic guitar and shaker. The atmosphere is cozy.",
        f"This is a stadium anthem in {instance_word} {object_class}, a powerful chorus with driving drums and roaring guitars. The atmosphere is triumphant.",
        f"This is a desert road trip track in {instance_word} {object_class}, a dusty groove with slide guitar and tambourine. The atmosphere is adventurous.",
        f"This is a nightclub banger in {instance_word} {object_class}, a thumping dance track with heavy bass. The atmosphere is euphoric.",
        f"This is a winter lullaby in {instance_word} {object_class}, soft vocals over a music box and muted strings. The atmosphere is tender.",
        f"This is a campfire singalong in {instance_word} {object_class}, an easy tune with group harmonies. The atmosphere is warm and communal.",
        f"This is a cinematic score in {instance_word} {object_class}, an epic theme with swelling orchestra and timpani. The atmosphere is dramatic.",
        f"This is an underground cypher in {instance_word} {object_class}, a gritty loop with sharp hi-hats. The atmosphere is raw.",
        f"This is a sunrise meditation in {instance_word} {object_class}, airy synth pads and wind chimes. The atmosphere is serene.",
        f"This is a backyard block party in {instance_word} {object_class}, a funky jam with horns and hand percussion. The atmosphere is festive.",
        f"This is a candlelit ballad in {instance_word} {object_class}, an emotional vocal over delicate fingerpicked guitar. The atmosphere is heartfelt.",
    ]


def get_style_prompts(instance_word, object_class):
    instruments = [
        "piano", "acoustic guitar", "electric guitar", "violin", "saxophone",
        "trumpet", "drums", "bass guitar", "synthesizer", "flute",
        "cello", "harp", "organ", "clarinet", "harmonica",
        "accordion", "banjo", "ukulele", "marimba", "vibraphone",
        "tabla", "djembe", "sitar", "erhu",
        "kalimba", "steel drums", "bagpipes", "theremin",
    ]
    return [f"a {inst} performance in {instance_word} {object_class}" for inst in instruments]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pretrained_base(device):
    path = PRETRAINED_PATH
    vae_config  = json.load(open(f"{path}/configs/vae_config.json"))
    stft_config = json.load(open(f"{path}/configs/stft_config.json"))
    vae  = AutoencoderKL(**vae_config).to(device)
    stft = TacotronSTFT(**stft_config).to(device)
    vae.load_state_dict(torch.load(f"{path}/vae/pytorch_model_vae.bin",   map_location=device))
    stft.load_state_dict(torch.load(f"{path}/stft/pytorch_model_stft.bin", map_location=device))
    vae.eval(); stft.eval()
    mfp = MusicFeaturePredictor(path=path, device=device)
    return vae, stft, mfp


def load_finetuned_model(model_dir, device):
    path = PRETRAINED_PATH
    main_config = json.load(open(f"{path}/configs/main_config.json"))
    model = MusicAudioDiffusion(
        text_encoder_name=main_config["text_encoder_name"],
        scheduler_name=main_config["scheduler_name"],
        unet_model_config_path=f"{path}/configs/music_diffusion_model_config.json",
        uncondition=True,
    ).to(device)
    ckpt = os.path.join(model_dir, "pytorch_model.bin")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")
    return model, scheduler


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, vae, mfp, scheduler, prompt, out_path_prefix, device):
    existing = [f"{out_path_prefix}_s{j+1}.wav" for j in range(NUM_WAVEFORMS)]
    if all(os.path.exists(p) for p in existing):
        print(f"  Skipped (already exists): {out_path_prefix}", flush=True)
        return existing

    beats, chords, chords_time = mfp.generate(prompt)
    paths = []
    for j in range(NUM_WAVEFORMS):
        latents = model.inference(
            [prompt], beats, [chords], [chords_time], scheduler,
            num_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE,
            num_samples_per_prompt=1, disable_progress=True,
        )
        mel = vae.decode_first_stage(latents)
        wav = vae.decode_to_waveform(mel)
        out_path = f"{out_path_prefix}_s{j+1}.wav"
        write(out_path, SAMPLE_RATE, wav[0])
        print(f"  Saved: {out_path}", flush=True)
        paths.append(out_path)

    gc.collect()
    torch.cuda.empty_cache()
    return paths


def process_genre(genre, vae, stft, mfp, device):
    model_dir = os.path.join(MODEL_DIR, genre, f"model_step_{MODEL_STEP}")
    out_dir   = os.path.join(OUT_BASE, genre)

    cfg_path = os.path.join(MODEL_DIR, genre, "class_name.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    instance_word = cfg["instance_word"]
    object_class  = cfg["object_class"]

    print(f"\nLoading model: {model_dir}", flush=True)
    model, scheduler = load_finetuned_model(model_dir, device)
    records = []

    ctx_dir = os.path.join(out_dir, "contextual")
    os.makedirs(ctx_dir, exist_ok=True)
    print("[Contextual prompts]", flush=True)
    for i, prompt in enumerate(get_contextual_prompts(instance_word, object_class)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug  = f"ctx_{i+1:02d}"
        paths = generate(model, vae, mfp, scheduler, prompt, os.path.join(ctx_dir, slug), device)
        for p in paths:
            records.append({"genre": genre, "object_class": object_class,
                             "prompt_type": "contextual", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    style_dir = os.path.join(out_dir, "style")
    os.makedirs(style_dir, exist_ok=True)
    print("[Style prompts]", flush=True)
    for i, prompt in enumerate(get_style_prompts(instance_word, object_class)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug  = f"style_{i+1:02d}"
        paths = generate(model, vae, mfp, scheduler, prompt, os.path.join(style_dir, slug), device)
        for p in paths:
            records.append({"genre": genre, "object_class": object_class,
                             "prompt_type": "style", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    del model
    torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    genres = list_genres()
    print(f"Found {len(genres)} genres with model_step_{MODEL_STEP}", flush=True)

    print("Loading base models (VAE, STFT, MFP)...", flush=True)
    vae, stft, mfp = load_pretrained_base(device)

    all_records = []
    for genre in genres:
        print(f"\n{'='*50}", flush=True)
        print(f"Genre: {genre}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_genre(genre, vae, stft, mfp, device))
        except Exception as e:
            import traceback
            print(f"ERROR on {genre}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv}", flush=True)
