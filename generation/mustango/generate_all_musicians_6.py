"""
Batch generate audio for all musician DreamBooth models (Mustango).
Run with: nohup python generate_all_musicians.py > generate_all_musicians.log 2>&1 &
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
MODEL_DIR = os.path.join(BASE, "composer_models_6")
OUT_BASE  = os.path.join(BASE, "generated_samples_musicians_6")

MODEL_STEP     = 1500   # matches train_all_musicians.sh (--max_train_steps=1500)
NUM_STEPS      = 100
GUIDANCE_SCALE = 3.0
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000


def list_musicians():
    """Return sorted composer_NNN_mustango_db folders that have the final step."""
    out = []
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.startswith("composer_") or not name.endswith("_mustango_db"):
            continue
        if os.path.isdir(os.path.join(MODEL_DIR, name, f"model_step_{MODEL_STEP}")):
            out.append(name)
    return out


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    """15 contextual prompts — Mustango descriptive style, musician as author/performer."""
    return [
        f"This is a live concert recording. {instance_word} {object_class} headlines the stage with a cheering crowd. The atmosphere is electrifying.",
        f"This is a late-night studio session. {instance_word} {object_class} lays down a slow groove over warm Rhodes keys. The atmosphere is intimate.",
        f"This is a rooftop sunset set. {instance_word} {object_class} drops a smooth downtempo track with soft percussion. The atmosphere is mellow.",
        f"This is a rainy-day session. {instance_word} {object_class} performs a reflective piece with soft piano. The atmosphere is melancholic.",
        f"This is a morning cafe session. {instance_word} {object_class} plays a light bossa nova with acoustic guitar and shaker. The atmosphere is cozy.",
        f"This is a stadium anthem. {instance_word} {object_class} leads a powerful chorus with driving drums. The atmosphere is triumphant.",
        f"This is a desert road trip track. {instance_word} {object_class} rolls out a dusty groove with slide guitar. The atmosphere is adventurous.",
        f"This is a nightclub banger. {instance_word} {object_class} drops a thumping dance track with heavy bass. The atmosphere is euphoric.",
        f"This is a winter lullaby. {instance_word} {object_class} sings softly over a music box and muted strings. The atmosphere is tender.",
        f"This is a campfire singalong. {instance_word} {object_class} strums an easy folk tune with group harmonies. The atmosphere is warm.",
        f"This is a cinematic score. {instance_word} {object_class} builds an epic theme with swelling orchestra. The atmosphere is dramatic.",
        f"This is an underground cypher. {instance_word} {object_class} rides a gritty boom-bap loop with sharp hi-hats. The atmosphere is raw.",
        f"This is a sunrise meditation. {instance_word} {object_class} floats across airy synth pads and wind chimes. The atmosphere is serene.",
        f"This is a backyard block party. {instance_word} {object_class} fires up a funky jam with horns. The atmosphere is festive.",
        f"This is a candlelit ballad. {instance_word} {object_class} delivers an emotional vocal over fingerpicked guitar. The atmosphere is heartfelt.",
    ]


def get_style_prompts(instance_word, object_class):
    styles = [
        "pop", "rock", "hip hop", "R&B", "jazz",
        "classical", "blues", "country", "electronic", "soul",
        "funk", "disco", "reggae", "metal", "folk",
        "Latin", "indie", "dance", "gospel", "trap",
        "punk", "K-pop", "lo-fi", "alternative",
        "house", "techno", "reggaeton", "ambient",
    ]
    return [f"a {style} song by {instance_word} {object_class}" for style in styles]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pretrained_base(device):
    path = PRETRAINED_PATH
    vae_config  = json.load(open(f"{path}/configs/vae_config.json"))
    stft_config = json.load(open(f"{path}/configs/stft_config.json"))

    vae  = AutoencoderKL(**vae_config).to(device)
    stft = TacotronSTFT(**stft_config).to(device)
    vae.load_state_dict(torch.load(f"{path}/vae/pytorch_model_vae.bin",   map_location=device))
    stft.load_state_dict(torch.load(f"{path}/stft/pytorch_model_stft.bin", map_location=device))
    vae.eval()
    stft.eval()

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

    scheduler = DDPMScheduler.from_pretrained(
        main_config["scheduler_name"], subfolder="scheduler"
    )
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
            [prompt],
            beats, [chords], [chords_time],
            scheduler,
            num_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            num_samples_per_prompt=1,
            disable_progress=True,
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


# ── Per-musician processing ───────────────────────────────────────────────────

def process_musician(musician, vae, stft, mfp, device):
    model_dir = os.path.join(MODEL_DIR, musician, f"model_step_{MODEL_STEP}")
    out_dir   = os.path.join(OUT_BASE, musician)

    cfg_path = os.path.join(MODEL_DIR, musician, "class_name.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    instance_word = cfg["instance_word"]
    object_class  = cfg["object_class"]

    ctx_prompts   = get_contextual_prompts(instance_word, object_class)
    style_prompts = get_style_prompts(instance_word, object_class)

    def all_done(sub, prompts, tag):
        sub_dir = os.path.join(out_dir, sub)
        for i in range(len(prompts)):
            slug = f"{tag}_{i+1:02d}"
            prefix = os.path.join(sub_dir, slug)
            for j in range(NUM_WAVEFORMS):
                if not os.path.exists(f"{prefix}_s{j+1}.wav"):
                    return False
        return True

    if all_done("contextual", ctx_prompts, "ctx") and all_done("style", style_prompts, "style"):
        print(f"  All outputs already exist for {musician}, skipping model load.", flush=True)
        records = []
        for i, prompt in enumerate(ctx_prompts):
            slug = f"ctx_{i+1:02d}"
            for j in range(NUM_WAVEFORMS):
                p = os.path.join(out_dir, "contextual", f"{slug}_s{j+1}.wav")
                records.append({
                    "musician": musician, "object_class": object_class,
                    "prompt_type": "contextual", "prompt_id": slug,
                    "prompt": prompt, "file": os.path.relpath(p, OUT_BASE),
                })
        for i, prompt in enumerate(style_prompts):
            slug = f"style_{i+1:02d}"
            for j in range(NUM_WAVEFORMS):
                p = os.path.join(out_dir, "style", f"{slug}_s{j+1}.wav")
                records.append({
                    "musician": musician, "object_class": object_class,
                    "prompt_type": "style", "prompt_id": slug,
                    "prompt": prompt, "file": os.path.relpath(p, OUT_BASE),
                })
        csv_path = os.path.join(out_dir, "metadata.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["musician", "object_class",
                                                    "prompt_type", "prompt_id",
                                                    "prompt", "file"])
            writer.writeheader()
            writer.writerows(records)
        return records

    print(f"\nLoading model: {model_dir}", flush=True)
    model, scheduler = load_finetuned_model(model_dir, device)

    records = []

    # Contextual prompts
    ctx_dir = os.path.join(out_dir, "contextual")
    os.makedirs(ctx_dir, exist_ok=True)
    print("[Contextual prompts]", flush=True)
    for i, prompt in enumerate(ctx_prompts):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug  = f"ctx_{i+1:02d}"
        paths = generate(model, vae, mfp, scheduler, prompt, os.path.join(ctx_dir, slug), device)
        for p in paths:
            records.append({
                "musician":     musician,
                "object_class": object_class,
                "prompt_type":  "contextual",
                "prompt_id":    slug,
                "prompt":       prompt,
                "file":         os.path.relpath(p, OUT_BASE),
            })

    # Style prompts
    style_dir = os.path.join(out_dir, "style")
    os.makedirs(style_dir, exist_ok=True)
    print("[Style prompts]", flush=True)
    for i, prompt in enumerate(style_prompts):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug  = f"style_{i+1:02d}"
        paths = generate(model, vae, mfp, scheduler, prompt, os.path.join(style_dir, slug), device)
        for p in paths:
            records.append({
                "musician":     musician,
                "object_class": object_class,
                "prompt_type":  "style",
                "prompt_id":    slug,
                "prompt":       prompt,
                "file":         os.path.relpath(p, OUT_BASE),
            })

    del model
    torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["musician", "object_class",
                                                "prompt_type", "prompt_id",
                                                "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    musicians = list_musicians()
    print(f"Found {len(musicians)} musicians with model_step_{MODEL_STEP}", flush=True)

    print("Loading base models (VAE, STFT, MFP)...", flush=True)
    vae, stft, mfp = load_pretrained_base(device)

    all_records = []
    for musician in musicians:
        print(f"\n{'='*50}", flush=True)
        print(f"Musician: {musician}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_musician(musician, vae, stft, mfp, device))
        except Exception as e:
            import traceback
            print(f"ERROR on {musician}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["musician", "object_class",
                                                "prompt_type", "prompt_id",
                                                "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv}", flush=True)
