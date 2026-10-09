"""
Batch generate audio for all composer_models_3 (3-clip, 1000-step, AudioLDM2).
Run with: nohup python generate_all_musicians_3.py > generate_all_musicians_3.log 2>&1 &
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
MODEL_DIR = os.path.join(BASE, "composer_models_3")
OUT_BASE  = os.path.join(BASE, "generated_samples_musicians_3")

PIPELINE_STEP  = 1000
NUM_STEPS      = 200
AUDIO_LENGTH   = 10
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000


def list_musicians():
    out = []
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.endswith("_ldm2_db"):
            continue
        if os.path.isdir(os.path.join(MODEL_DIR, name, f"pipeline_step_{PIPELINE_STEP}")):
            out.append(name)
    return out


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_contextual_prompts(instance_word, object_class):
    return [
        f"A {instance_word} {object_class} performing a heartfelt set over gentle piano chords.",
        f"A {instance_word} {object_class} laying down a late-night groove with warm Rhodes keys.",
        f"A {instance_word} {object_class} playing smooth downtempo on a rooftop at sunset.",
        f"A {instance_word} {object_class} performing a reflective piece with soft piano on a rainy day.",
        f"A {instance_word} {object_class} playing a light bossa nova with acoustic guitar in a morning cafe.",
        f"A {instance_word} {object_class} leading a powerful stadium anthem with driving drums.",
        f"A {instance_word} {object_class} rolling out a dusty groove with slide guitar.",
        f"A {instance_word} {object_class} dropping a thumping nightclub banger with heavy bass.",
        f"A {instance_word} {object_class} singing softly over a music box and muted strings.",
        f"A {instance_word} {object_class} strumming an easy folk tune by a campfire with group harmonies.",
        f"A {instance_word} {object_class} building an epic cinematic theme with swelling orchestra.",
        f"A {instance_word} {object_class} riding a gritty boom-bap loop with sharp hi-hats.",
        f"A {instance_word} {object_class} floating across airy synth pads and wind chimes.",
        f"A {instance_word} {object_class} firing up a funky jam with horns and hand percussion.",
        f"A {instance_word} {object_class} delivering an emotional ballad over delicate fingerpicked guitar.",
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


def process_musician(musician):
    model_dir = os.path.join(MODEL_DIR, musician)
    out_dir   = os.path.join(OUT_BASE, musician)

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
            records.append({"musician": musician, "object_class": object_class,
                             "prompt_type": "contextual", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    style_dir = os.path.join(out_dir, "style")
    os.makedirs(style_dir, exist_ok=True)
    print("[Style prompts]", flush=True)
    for i, prompt in enumerate(get_style_prompts(instance_word, object_class)):
        print(f"  [{i+1}] {prompt}", flush=True)
        slug = f"style_{i+1:02d}"
        paths = generate(pipe, prompt, os.path.join(style_dir, slug))
        for p in paths:
            records.append({"musician": musician, "object_class": object_class,
                             "prompt_type": "style", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    del pipe
    torch.cuda.empty_cache()

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["musician", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Metadata: {csv_path}", flush=True)
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    musicians = list_musicians()
    print(f"Found {len(musicians)} musicians with pipeline_step_{PIPELINE_STEP}", flush=True)

    all_records = []
    for musician in musicians:
        print(f"\n{'='*50}", flush=True)
        print(f"Musician: {musician}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_musician(musician))
        except Exception as e:
            import traceback
            print(f"ERROR on {musician}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["musician", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
