"""
Batch generate audio for all instruments using Stable Audio dreambooth checkpoints.
Uses detailed prompts with genre context, BPM, and arrangement details
(generate_bagpipe.py style).

Run: nohup python generate_all.py > generate_all.log 2>&1 &
"""

import gc
import torch
import os
import sys
import csv
from einops import rearrange
from scipy.io.wavfile import write

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# ── Config ────────────────────────────────────────────────────────────────────

BASE      = "/path/to/Data-attribution/stable-audio-tools"
MODEL_DIR = os.path.join(BASE, "instrument_model_6")
OUT_BASE  = os.path.join(BASE, "generated_samples_instrument_6")

INSTRUMENTS = [
    "violin", "cello", "acoustic_guitar", "electric_guitar", "bass_guitar",
    "harp", "flute", "clarinet", "oboe", "saxophone",
    "bassoon", "trumpet", "trombone", "french_horn", "tuba",
    "piano", "organ", "accordion", "drums", "xylophone",
    "marimba", "timpani", "harmonica", "banjo", "ukulele",
]

CKPT_STEP      = 1500
INSTANCE_WORD = "sks"
NUM_STEPS = 200
CFG_SCALE = 4.0
NUM_WAVEFORMS = 4
WINDOW_SECONDS = 10
SAMPLER_TYPE = "dpmpp-3m-sde"
SIGMA_MIN = 0.3
SIGMA_MAX = 500.0

# ── Prompts (generate_bagpipe.py style) ──────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    """High-frequency FMA genres as context, style first, instrument after."""
    return [
        # Rock (622 tracks in training data) - strongest prior
        f"Driving rock instrumental, electric guitar riff, 120 BPM, featuring {instance_word} {object_class}",
        f"Heavy rock anthem, distorted power chords, 130 BPM, featuring {instance_word} {object_class}",
        # Electronic (557 tracks)
        f"Atmospheric electronic ambient, synthesizer pads, 110 BPM, featuring {instance_word} {object_class}",
        f"Energetic electronic dance track, pulsing beat, 128 BPM, featuring {instance_word} {object_class}",
        # Pop (388 tracks)
        f"Upbeat pop instrumental, catchy melody, 115 BPM, featuring {instance_word} {object_class}",
        f"Bright pop ballad, piano and vocals, 95 BPM, featuring {instance_word} {object_class}",
        # Experimental (386 tracks)
        f"Experimental ambient soundscape, textured drone, 90 BPM, featuring {instance_word} {object_class}",
        # Hip-Hop (335 tracks)
        f"Hard hitting hip hop beat, deep bass, 90 BPM, featuring {instance_word} {object_class}",
        f"Laid back lo-fi hip hop, vinyl crackle, 85 BPM, featuring {instance_word} {object_class}",
        # Classical (272 tracks)
        f"Orchestral classical piece, strings and woodwinds, 80 BPM, featuring {instance_word} {object_class}",
        f"Dramatic classical crescendo, full orchestra, 100 BPM, featuring {instance_word} {object_class}",
        # Instrumental (235 tracks)
        f"Warm instrumental acoustic melody, fingerpicked guitar, 100 BPM, featuring {instance_word} {object_class}",
        # Folk (160 tracks)
        f"Traditional folk tune, acoustic ensemble, 105 BPM, featuring {instance_word} {object_class}",
        # Jazz (145 tracks)
        f"Smooth jazz groove, walking bass line, 110 BPM, featuring {instance_word} {object_class}",
        # Country (109 tracks)
        f"Country acoustic ballad, steel guitar, 95 BPM, featuring {instance_word} {object_class}",
    ]


def get_style_prompts(instance_word, object_class):
    """Only FMA high-frequency genres (>100 tracks in training data)."""
    styles = [
        ("rock", "120 BPM"),
        ("electronic", "128 BPM"),
        ("pop", "115 BPM"),
        ("experimental", "95 BPM"),
        ("hip hop", "90 BPM"),
        ("classical", "80 BPM"),
        ("instrumental", "100 BPM"),
        ("folk", "105 BPM"),
    ]
    return [
        f"Energetic {style} track, rich arrangement, {bpm}, featuring {instance_word} {object_class}"
        for style, bpm in styles
    ]


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_config


# ── Inference ────────────────────────────────────────────────────────────────

def save_audio(audio, out_path, sample_rate):
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.float().div(torch.max(torch.abs(audio))).clamp(-1, 1)
    audio_np = audio.cpu().numpy().T
    write(out_path, sample_rate, audio_np)


@torch.no_grad()
def generate(model, model_config, prompt, out_path_prefix, device):
    sample_rate = model_config["sample_rate"]
    sample_size = sample_rate * WINDOW_SECONDS

    # Skip if all waveforms already exist (resume support)
    existing = [f"{out_path_prefix}_s{j+1}.wav" for j in range(NUM_WAVEFORMS)]
    if all(os.path.exists(p) for p in existing):
        print(f"  Skipped (already exists): {out_path_prefix}", flush=True)
        return existing

    paths = []
    for j in range(NUM_WAVEFORMS):
        audio = generate_diffusion_cond(
            model,
            steps=NUM_STEPS,
            cfg_scale=CFG_SCALE,
            conditioning=[{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": WINDOW_SECONDS,
            }],
            sample_size=sample_size,
            device=device,
            sampler_type=SAMPLER_TYPE,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
        )
        out_path = f"{out_path_prefix}_s{j+1}.wav"
        save_audio(audio, out_path, sample_rate)
        print(f"  Saved: {out_path}", flush=True)
        paths.append(out_path)

    # Free GPU memory between prompts
    gc.collect()
    torch.cuda.empty_cache()

    return paths


# ── Per-instrument processing ────────────────────────────────────────────────

def process_instrument(instrument, device):
    ckpt_dir = os.path.join(MODEL_DIR, f"{instrument}_sa_db")
    ckpt_path = os.path.join(ckpt_dir, f"model_step_{CKPT_STEP}", "pytorch_model.ckpt")

    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}, skipping", flush=True)
        return []

    print(f"\nLoading model: {ckpt_path}", flush=True)
    model, model_config = load_model(ckpt_path, device)
    out_dir = os.path.join(OUT_BASE, f"{instrument}_sa_db")
    records = []

    # Contextual prompts
    ctx_dir = os.path.join(out_dir, "contextual")
    os.makedirs(ctx_dir, exist_ok=True)
    print("[Contextual prompts]", flush=True)
    for i, prompt in enumerate(get_contextual_prompts(INSTANCE_WORD, instrument)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"ctx_{i+1:02d}"
        paths = generate(model, model_config, prompt, os.path.join(ctx_dir, slug), device)
        for p in paths:
            records.append({
                "instrument": instrument,
                "step": CKPT_STEP,
                "prompt_type": "contextual",
                "prompt_id": slug,
                "prompt": prompt,
                "file": os.path.relpath(p, OUT_BASE),
            })

    # Style prompts
    style_dir = os.path.join(out_dir, "style")
    os.makedirs(style_dir, exist_ok=True)
    print("[Style prompts]", flush=True)
    for i, prompt in enumerate(get_style_prompts(INSTANCE_WORD, instrument)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"style_{i+1:02d}"
        paths = generate(model, model_config, prompt, os.path.join(style_dir, slug), device)
        for p in paths:
            records.append({
                "instrument": instrument,
                "step": CKPT_STEP,
                "prompt_type": "style",
                "prompt_id": slug,
                "prompt": prompt,
                "file": os.path.relpath(p, OUT_BASE),
            })

    # Per-instrument metadata CSV
    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "step", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return records


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_records = []

    for instrument in INSTRUMENTS:
        print(f"\n{'='*50}", flush=True)
        print(f"Instrument: {instrument}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_instrument(instrument, device))
        except Exception as e:
            import traceback
            print(f"ERROR on {instrument}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    # Write combined metadata CSV
    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all_sa.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "step", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
