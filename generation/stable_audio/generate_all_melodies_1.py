"""
Batch generate audio for all melody_models_1 (1-clip, 300-step, Stable Audio).
Run with: nohup python generate_all_melodies.py > generate_all_melodies.log 2>&1 &
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
MODEL_DIR = os.path.join(BASE, "melody_models_1")
OUT_BASE  = os.path.join(BASE, "generated_samples_melodies_1")

CKPT_STEP      = 300
INSTANCE_WORD  = "sks"
OBJECT_CLASS   = "melody"
NUM_STEPS      = 200
CFG_SCALE      = 4.0
NUM_WAVEFORMS  = 4
WINDOW_SECONDS = 10
SAMPLER_TYPE   = "dpmpp-3m-sde"
SIGMA_MIN      = 0.3
SIGMA_MAX      = 500.0


def list_melodies():
    out = []
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.startswith("melody_") or not name.endswith("_sa_db"):
            continue
        ckpt = os.path.join(MODEL_DIR, name, f"model_step_{CKPT_STEP}", "pytorch_model.ckpt")
        if os.path.exists(ckpt):
            out.append(name)
    return out


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    return [
        f"Piano solo, {instance_word} {object_class}, expressive, 80 BPM",
        f"String quartet arrangement, {instance_word} {object_class}, 75 BPM",
        f"Orchestral version, full ensemble, {instance_word} {object_class}, 90 BPM",
        f"Acoustic guitar fingerpicking, {instance_word} {object_class}, 85 BPM",
        f"Jazz reharmonization, piano and bass, {instance_word} {object_class}, 100 BPM",
        f"Electronic ambient version, synth pads, {instance_word} {object_class}, 70 BPM",
        f"Lo-fi hip hop beat, vinyl texture, {instance_word} {object_class}, 80 BPM",
        f"Cinematic orchestral, emotional strings, {instance_word} {object_class}, 65 BPM",
        f"Upbeat pop arrangement, {instance_word} {object_class}, 115 BPM",
        f"Minimalist piano version, {instance_word} {object_class}, 60 BPM",
        f"Flute and harp duet, {instance_word} {object_class}, 75 BPM",
        f"Electric guitar lead, band arrangement, {instance_word} {object_class}, 110 BPM",
        f"Choir and organ, {instance_word} {object_class}, 70 BPM",
        f"Bossa nova style, nylon guitar and percussion, {instance_word} {object_class}, 95 BPM",
        f"Marimba and vibraphone, {instance_word} {object_class}, 100 BPM",
    ]


def get_style_prompts(instance_word, object_class):
    styles = [
        ("classical", "80 BPM"), ("jazz", "100 BPM"), ("pop", "115 BPM"),
        ("electronic", "110 BPM"), ("folk", "95 BPM"), ("ambient", "65 BPM"),
        ("lo-fi", "75 BPM"), ("cinematic", "85 BPM"),
    ]
    return [f"{instance_word} {object_class}, {style} arrangement, {bpm}"
            for style, bpm in styles]


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


def process_melody(melody, device):
    ckpt_path = os.path.join(MODEL_DIR, melody, f"model_step_{CKPT_STEP}", "pytorch_model.ckpt")
    out_dir   = os.path.join(OUT_BASE, melody)

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
            records.append({"melody": melody, "step": CKPT_STEP,
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
            records.append({"melody": melody, "step": CKPT_STEP,
                             "prompt_type": "style", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["melody", "step", "prompt_type", "prompt_id", "prompt", "file"])
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

    melodies = list_melodies()
    print(f"Found {len(melodies)} melodies with model_step_{CKPT_STEP}", flush=True)

    all_records = []
    for melody in melodies:
        print(f"\n{'='*50}", flush=True)
        print(f"Melody: {melody}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_melody(melody, device))
        except Exception as e:
            import traceback
            print(f"ERROR on {melody}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all_sa.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["melody", "step", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
