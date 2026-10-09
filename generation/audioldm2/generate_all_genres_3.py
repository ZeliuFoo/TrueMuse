"""
Batch generate audio for all genre DreamBooth models (AudioLDM2).
Run with: nohup python generate_all_genres.py > generate_all_genres.log 2>&1 &
"""

import gc
import os
import sys
import csv
import json
import torch
from scipy.io.wavfile import write
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline

# ── Config ────────────────────────────────────────────────────────────────────

BASE      = "/path/to/Data-attribution/"
MODEL_DIR = os.path.join(BASE, "genre_models_3")
OUT_BASE  = os.path.join(BASE, "generated_samples_genres_3")

PIPELINE_STEP  = 1000
NUM_STEPS      = 200
AUDIO_LENGTH   = 10
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000


def list_genres():
    """Return sorted list of <genre>_ldm2_db folders that have the final pipeline."""
    out = []
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.endswith("_ldm2_db"):
            continue
        if os.path.isdir(os.path.join(MODEL_DIR, name,
                                      f"pipeline_step_{PIPELINE_STEP}")):
            out.append(name)
    return out


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    return [
        f"A driving instrumental in {instance_word} {object_class} with electric guitar riff.",
        f"A heavy anthem in {instance_word} {object_class} with distorted power chords.",
        f"An atmospheric soundscape in {instance_word} {object_class} with synthesizer pads.",
        f"An energetic dance track in {instance_word} {object_class} with a pulsing beat.",
        f"An upbeat melodic tune in {instance_word} {object_class} with a catchy hook.",
        f"A bright ballad in {instance_word} {object_class} with piano and vocals.",
        f"An experimental ambient piece in {instance_word} {object_class} with textured drones.",
        f"A hard-hitting track in {instance_word} {object_class} with deep bass.",
        f"A laid-back lo-fi loop in {instance_word} {object_class} with vinyl crackle.",
        f"An orchestral piece in {instance_word} {object_class} with strings and woodwinds.",
        f"A dramatic crescendo in {instance_word} {object_class} with full orchestra.",
        f"A warm acoustic melody in {instance_word} {object_class} with fingerpicked guitar.",
        f"A traditional tune in {instance_word} {object_class} played by an acoustic ensemble.",
        f"A smooth groove in {instance_word} {object_class} with a walking bass line.",
        f"An acoustic ballad in {instance_word} {object_class} with steel guitar.",
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


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(pipe, prompt, out_path_prefix):
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

    gc.collect()
    torch.cuda.empty_cache()
    return paths


def process_genre(genre):
    model_dir = os.path.join(MODEL_DIR, genre)
    out_dir   = os.path.join(OUT_BASE, genre)

    with open(os.path.join(model_dir, "class_name.json")) as f:
        cfg = json.load(f)
    instance_word = cfg["instance_word"]
    object_class  = cfg["object_class"]

    pipeline_path = os.path.join(model_dir, f"pipeline_step_{PIPELINE_STEP}")
    print(f"\nLoading pipeline: {pipeline_path}", flush=True)
    pipe = AudioLDM2Pipeline.from_pretrained(pipeline_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    records = []

    ctx_dir = os.path.join(out_dir, "contextual")
    os.makedirs(ctx_dir, exist_ok=True)
    print("[Contextual prompts]", flush=True)
    for i, prompt in enumerate(get_contextual_prompts(instance_word, object_class)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"ctx_{i+1:02d}"
        paths = generate(pipe, prompt, os.path.join(ctx_dir, slug))
        for p in paths:
            records.append({
                "genre": genre,
                "object_class": object_class,
                "prompt_type": "contextual",
                "prompt_id": slug,
                "prompt": prompt,
                "file": os.path.relpath(p, OUT_BASE),
            })

    style_dir = os.path.join(out_dir, "style")
    os.makedirs(style_dir, exist_ok=True)
    print("[Style prompts]", flush=True)
    for i, prompt in enumerate(get_style_prompts(instance_word, object_class)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"style_{i+1:02d}"
        paths = generate(pipe, prompt, os.path.join(style_dir, slug))
        for p in paths:
            records.append({
                "genre": genre,
                "object_class": object_class,
                "prompt_type": "style",
                "prompt_id": slug,
                "prompt": prompt,
                "file": os.path.relpath(p, OUT_BASE),
            })

    del pipe
    torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "object_class",
                                                "prompt_type", "prompt_id",
                                                "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    genres = list_genres()
    print(f"Found {len(genres)} genres with pipeline_step_{PIPELINE_STEP}",
          flush=True)

    all_records = []
    for genre in genres:
        print(f"\n{'='*50}", flush=True)
        print(f"Genre: {genre}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_genre(genre))
        except Exception as e:
            print(f"ERROR on {genre}: {e}", flush=True, file=sys.stderr)
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "object_class",
                                                "prompt_type", "prompt_id",
                                                "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
