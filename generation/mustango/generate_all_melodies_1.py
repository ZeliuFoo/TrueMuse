"""
Batch generate audio for all melody_models_1 (1-clip, 300-step, Mustango).
Run with: nohup python generate_all_melodies_1.py > generate_all_melodies_1.log 2>&1 &
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
MODEL_DIR = os.path.join(BASE, "melody_models_1")
OUT_BASE  = os.path.join(BASE, "generated_samples_melodies_1")

MODEL_STEP     = 300
NUM_STEPS      = 100
GUIDANCE_SCALE = 3.0
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000


def list_melodies():
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
        f"This is a piano solo rendition of {instance_word} {object_class}. The performance is expressive and lyrical.",
        f"This is a string quartet arrangement of {instance_word} {object_class}. The atmosphere is warm and intimate.",
        f"This is an orchestral version of {instance_word} {object_class} with full ensemble. The atmosphere is grand.",
        f"This is an acoustic guitar fingerpicking rendition of {instance_word} {object_class}. The atmosphere is gentle.",
        f"This is a jazz reharmonization of {instance_word} {object_class} with piano and upright bass. The atmosphere is cool.",
        f"This is an electronic ambient version of {instance_word} {object_class} with lush synth pads. The atmosphere is ethereal.",
        f"This is a lo-fi hip hop beat built around {instance_word} {object_class} with vinyl crackle. The atmosphere is relaxed.",
        f"This is a cinematic orchestral rendering of {instance_word} {object_class} with emotional strings. The atmosphere is dramatic.",
        f"This is an upbeat pop arrangement of {instance_word} {object_class}. The atmosphere is energetic.",
        f"This is a minimalist piano version of {instance_word} {object_class}. The atmosphere is contemplative.",
        f"This is a flute and harp duet playing {instance_word} {object_class}. The atmosphere is delicate.",
        f"This is an electric guitar lead rendition of {instance_word} {object_class} with full band. The atmosphere is powerful.",
        f"This is a choir and organ performance of {instance_word} {object_class}. The atmosphere is solemn.",
        f"This is a bossa nova arrangement of {instance_word} {object_class} with nylon guitar and light percussion. The atmosphere is breezy.",
        f"This is a marimba and vibraphone rendition of {instance_word} {object_class}. The atmosphere is playful.",
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
    return [f"a {style} arrangement of {instance_word} {object_class}" for style in styles]


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


def process_melody(melody, vae, stft, mfp, device):
    model_dir = os.path.join(MODEL_DIR, melody, f"model_step_{MODEL_STEP}")
    out_dir   = os.path.join(OUT_BASE, melody)

    cfg_path = os.path.join(MODEL_DIR, melody, "class_name.json")
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
            records.append({"melody": melody, "object_class": object_class,
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
            records.append({"melody": melody, "object_class": object_class,
                             "prompt_type": "style", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    del model
    torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["melody", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    melodies = list_melodies()
    print(f"Found {len(melodies)} melodies with model_step_{MODEL_STEP}", flush=True)

    print("Loading base models (VAE, STFT, MFP)...", flush=True)
    vae, stft, mfp = load_pretrained_base(device)

    all_records = []
    for melody in melodies:
        print(f"\n{'='*50}", flush=True)
        print(f"Melody: {melody}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_melody(melody, vae, stft, mfp, device))
        except Exception as e:
            import traceback
            print(f"ERROR on {melody}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["melody", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv}", flush=True)
