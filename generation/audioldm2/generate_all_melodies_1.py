"""
Batch generate audio for all melody_models_1 (1-clip, 300-step, AudioLDM2).
Run with: nohup python generate_all_melodies_1.py > generate_all_melodies_1.log 2>&1 &
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
MODEL_DIR = os.path.join(BASE, "melody_models_1")
OUT_BASE  = os.path.join(BASE, "generated_samples_melodies_1")

PIPELINE_STEP  = 300
NUM_STEPS      = 200
AUDIO_LENGTH   = 10
NUM_WAVEFORMS  = 4
SAMPLE_RATE    = 16000


def list_melodies():
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
        f"A piano solo performing {instance_word} {object_class}, expressive and lyrical.",
        f"A string quartet arrangement of {instance_word} {object_class}, warm and intimate.",
        f"An orchestral version of {instance_word} {object_class} with full ensemble.",
        f"An acoustic guitar fingerpicking rendition of {instance_word} {object_class}.",
        f"A jazz reharmonization of {instance_word} {object_class} with piano and bass.",
        f"An electronic ambient version of {instance_word} {object_class} with synth pads.",
        f"A lo-fi hip hop beat built around {instance_word} {object_class} with vinyl texture.",
        f"A cinematic orchestral rendering of {instance_word} {object_class} with emotional strings.",
        f"An upbeat pop arrangement of {instance_word} {object_class}.",
        f"A minimalist piano version of {instance_word} {object_class}.",
        f"A flute and harp duet playing {instance_word} {object_class}.",
        f"An electric guitar lead rendition of {instance_word} {object_class} with band.",
        f"A choir and organ performance of {instance_word} {object_class}.",
        f"A bossa nova arrangement of {instance_word} {object_class} with nylon guitar.",
        f"A marimba and vibraphone rendition of {instance_word} {object_class}.",
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


def process_melody(melody):
    model_dir = os.path.join(MODEL_DIR, melody)
    out_dir   = os.path.join(OUT_BASE, melody)

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
            records.append({"melody": melody, "object_class": object_class,
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
            records.append({"melody": melody, "object_class": object_class,
                             "prompt_type": "style", "prompt_id": slug,
                             "prompt": prompt, "file": os.path.relpath(p, OUT_BASE)})

    del pipe
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
    melodies = list_melodies()
    print(f"Found {len(melodies)} melodies with pipeline_step_{PIPELINE_STEP}", flush=True)

    all_records = []
    for melody in melodies:
        print(f"\n{'='*50}", flush=True)
        print(f"Melody: {melody}", flush=True)
        print(f"{'='*50}", flush=True)
        try:
            all_records.extend(process_melody(melody))
        except Exception as e:
            import traceback
            print(f"ERROR on {melody}: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            continue

    os.makedirs(OUT_BASE, exist_ok=True)
    all_csv_path = os.path.join(OUT_BASE, "metadata_all.csv")
    with open(all_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["melody", "object_class", "prompt_type", "prompt_id", "prompt", "file"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nAll done. Combined metadata: {all_csv_path}", flush=True)
