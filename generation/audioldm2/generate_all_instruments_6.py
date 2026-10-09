"""
Batch generate audio for all instruments using contextual and style prompts.
Run with: nohup python generate_all.py > generate_all.log 2>&1 &
"""

import gc
import torch
import json
import os
import sys
import csv
from scipy.io.wavfile import write
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline

# ── Config ────────────────────────────────────────────────────────────────────

BASE       = "/path/to/Data-attribution/"
MODEL_DIR  = os.path.join(BASE, "instrument_model_6")
OUT_BASE   = os.path.join(BASE, "generated_samples_instrument_6")

INSTRUMENTS = [
    "violin_ldm2_db",
    "cello_ldm2_db",
    "acoustic_guitar_ldm2_db",
    "electric_guitar_ldm2_db",
    "bass_guitar_ldm2_db",
    "harp_ldm2_db",
    "flute_ldm2_db",
    "clarinet_ldm2_db",
    "oboe_ldm2_db",
    "saxophone_ldm2_db",
    "bassoon_ldm2_db",
    "trumpet_ldm2_db",
    "trombone_ldm2_db",
    "french_horn_ldm2_db",
    "tuba_ldm2_db",
    "piano_ldm2_db",
    "organ_ldm2_db",
    "accordion_ldm2_db",
    "drums_ldm2_db",
    "xylophone_ldm2_db",
    "marimba_ldm2_db",
    "timpani_ldm2_db",
    "harmonica_ldm2_db",
    "banjo_ldm2_db",
    "ukulele_ldm2_db",
]

PIPELINE_STEP  = 1500
NUM_STEPS      = 200
AUDIO_LENGTH   = 10
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000

# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    # 15 prompts in AudioLDM2 official example style — natural music scene descriptions
    return [
        f"A {instance_word} {object_class} playing a heartfelt melody over gentle piano chords.",
        f"A {instance_word} {object_class} grooving in sync with drums and bass.",
        f"A catchy beat with {instance_word} {object_class} in the mix.",
        f"A {instance_word} {object_class} creating futuristic soundscapes.",
        f"A {instance_word} {object_class} playing a lively reel with hand claps.",
        f"A cheerful {instance_word} {object_class} strumming in a beachside jam.",
        f"A {instance_word} {object_class} solo over a smooth jazz backing track.",
        f"A {instance_word} {object_class} leading an energetic street parade.",
        f"A dreamy {instance_word} {object_class} layered with ambient pads and reverb.",
        f"A {instance_word} {object_class} duet with acoustic guitar by a campfire.",
        f"A {instance_word} {object_class} performing an emotional ballad with strings.",
        f"A powerful {instance_word} {object_class} ensemble, exuding power and precision.",
        f"A {instance_word} {object_class} improvising over a lo-fi hip hop beat.",
        f"A {instance_word} {object_class} blending with electronic synths in a chill mix.",
        f"A {instance_word} {object_class} echoing through a grand cathedral with choir.",
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
    return [f"a {style} song with {instance_word} {object_class}" for style in styles]


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(pipe, prompt, out_path_prefix):
    # Skip if all waveforms already exist (resume support)
    existing = [f"{out_path_prefix}_s{j+1}.wav" for j in range(NUM_WAVEFORMS)]
    if all(os.path.exists(p) for p in existing):
        print(f"  Skipped (already exists): {out_path_prefix}", flush=True)
        return existing

    audios = pipe(
        prompt,
        num_inference_steps=NUM_STEPS,
        num_waveforms_per_prompt=NUM_WAVEFORMS,
        audio_length_in_s=AUDIO_LENGTH,
    ).audios
    paths = []
    for j, waveform in enumerate(audios):
        out_path = f"{out_path_prefix}_s{j+1}.wav"
        write(out_path, SAMPLE_RATE, waveform)
        print(f"  Saved: {out_path}", flush=True)
        paths.append(out_path)

    # Free GPU memory between prompts to prevent CUDA OOM hang
    gc.collect()
    torch.cuda.empty_cache()

    return paths


def process_instrument(instrument):
    model_dir = os.path.join(MODEL_DIR, instrument)
    out_dir   = os.path.join(OUT_BASE, instrument)

    with open(os.path.join(model_dir, "class_name.json")) as f:
        cfg = json.load(f)
    instance_word = cfg["instance_word"]
    object_class  = cfg["object_class"]

    pipeline_path = os.path.join(model_dir, f"pipeline_step_{PIPELINE_STEP}")
    print(f"\nLoading pipeline: {pipeline_path}", flush=True)
    pipe = AudioLDM2Pipeline.from_pretrained(pipeline_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    records = []

    # Contextual prompts
    ctx_dir = os.path.join(out_dir, "contextual")
    os.makedirs(ctx_dir, exist_ok=True)
    print("[Contextual prompts]", flush=True)
    for i, prompt in enumerate(get_contextual_prompts(instance_word, object_class)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"ctx_{i+1:02d}"
        paths = generate(pipe, prompt, os.path.join(ctx_dir, slug))
        for p in paths:
            records.append({
                "instrument": instrument,
                "object_class": object_class,
                "prompt_type": "contextual",
                "prompt_id": slug,
                "prompt": prompt,
                "file": os.path.relpath(p, OUT_BASE),
            })

    # Style prompts
    style_dir = os.path.join(out_dir, "style")
    os.makedirs(style_dir, exist_ok=True)
    print("[Style prompts]", flush=True)
    for i, prompt in enumerate(get_style_prompts(instance_word, object_class)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"style_{i+1:02d}"
        paths = generate(pipe, prompt, os.path.join(style_dir, slug))
        for p in paths:
            records.append({
                "instrument": instrument,
                "object_class": object_class,
                "prompt_type": "style",
                "prompt_id": slug,
                "prompt": prompt,
                "file": os.path.relpath(p, OUT_BASE),
            })

    del pipe
    torch.cuda.empty_cache()

    # Write per-instrument metadata CSV
    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_records = []
    for instrument in INSTRUMENTS:
        print(f"\n{'='*50}", flush=True)
        print(f"Instrument: {instrument}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_instrument(instrument))
        except Exception as e:
            print(f"ERROR on {instrument}: {e}", flush=True, file=sys.stderr)
            continue

    # Write combined metadata CSV for all instruments
    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
