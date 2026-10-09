"""
Batch generate audio for all instruments using instrument_model_3 (3-clip, 1000-step, Mustango).
Run with: nohup python generate_all_instruments_3.py > generate_all_instruments_3.log 2>&1 &
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
MODEL_DIR = os.path.join(BASE, "instrument_model_3")
OUT_BASE  = os.path.join(BASE, "generated_samples_instrument_3")

MODEL_STEP     = 1000
NUM_STEPS      = 100
GUIDANCE_SCALE = 3.0
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000

INSTRUMENTS = [
    "violin_mustango_db", "cello_mustango_db", "acoustic_guitar_mustango_db",
    "electric_guitar_mustango_db", "bass_guitar_mustango_db", "harp_mustango_db",
    "flute_mustango_db", "clarinet_mustango_db", "oboe_mustango_db",
    "saxophone_mustango_db", "bassoon_mustango_db", "trumpet_mustango_db",
    "trombone_mustango_db", "french_horn_mustango_db", "tuba_mustango_db",
    "piano_mustango_db", "organ_mustango_db", "accordion_mustango_db",
    "drums_mustango_db", "xylophone_mustango_db", "marimba_mustango_db",
    "timpani_mustango_db", "harmonica_mustango_db", "banjo_mustango_db",
    "ukulele_mustango_db",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    return [
        f"This is a live concert recording. A {instance_word} {object_class} headlines the stage with a cheering crowd. The atmosphere is electrifying.",
        f"This is a late-night studio session. A {instance_word} {object_class} lays down a slow groove over warm Rhodes keys. The atmosphere is intimate.",
        f"This is a rooftop sunset set. A {instance_word} {object_class} plays smooth downtempo with soft percussion. The atmosphere is mellow.",
        f"This is a rainy-day session. A {instance_word} {object_class} performs a reflective piece with soft piano. The atmosphere is melancholic.",
        f"This is a morning cafe session. A {instance_word} {object_class} plays a light bossa nova with acoustic guitar and shaker. The atmosphere is cozy.",
        f"This is a stadium anthem. A {instance_word} {object_class} leads a powerful chorus with driving drums. The atmosphere is triumphant.",
        f"This is a desert road trip track. A {instance_word} {object_class} rolls out a dusty groove with tambourine. The atmosphere is adventurous.",
        f"This is a nightclub banger. A {instance_word} {object_class} anchors a thumping dance track with heavy bass. The atmosphere is euphoric.",
        f"This is a winter lullaby. A {instance_word} {object_class} plays softly over muted strings. The atmosphere is tender.",
        f"This is a campfire singalong. A {instance_word} {object_class} leads an easy folk tune with group harmonies. The atmosphere is warm.",
        f"This is a cinematic score. A {instance_word} {object_class} builds an epic theme with swelling orchestra. The atmosphere is dramatic.",
        f"This is an underground cypher. A {instance_word} {object_class} rides a gritty boom-bap loop with sharp hi-hats. The atmosphere is raw.",
        f"This is a sunrise meditation. A {instance_word} {object_class} floats across airy synth pads and wind chimes. The atmosphere is serene.",
        f"This is a backyard block party. A {instance_word} {object_class} fires up a funky jam with horns. The atmosphere is festive.",
        f"This is a candlelit ballad. A {instance_word} {object_class} delivers an emotional melody over fingerpicked guitar. The atmosphere is heartfelt.",
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
    return [
        f"This is a {style} song with a {instance_word} {object_class} playing the lead melody."
        for style in styles
    ]


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


def process_instrument(instrument, vae, stft, mfp, device):
    model_dir = os.path.join(MODEL_DIR, instrument, f"model_step_{MODEL_STEP}")
    out_dir   = os.path.join(OUT_BASE, instrument)

    cfg_path = os.path.join(MODEL_DIR, instrument, "class_name.json")
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
            records.append({"instrument": instrument, "object_class": object_class,
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
            records.append({"instrument": instrument, "object_class": object_class,
                             "prompt_type": "style", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    del model
    torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading base models (VAE, STFT, MFP)...", flush=True)
    vae, stft, mfp = load_pretrained_base(device)

    all_records = []
    for instrument in INSTRUMENTS:
        print(f"\n{'='*50}", flush=True)
        print(f"Instrument: {instrument}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_instrument(instrument, vae, stft, mfp, device))
        except Exception as e:
            import traceback
            print(f"ERROR on {instrument}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
