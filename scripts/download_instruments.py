"""Rebuild the instrument concept clips from YouTube.

The released dataset ships only data/concepts/metadata_instruments.csv for the instrument set
(YouTube ID + start time); the audio itself is not redistributed. This script downloads each
video's audio once and cuts `num_clips` consecutive `clip_duration_sec`-second segments starting
at `start_sec`, then writes

    <data_root>/concepts/instrument_6/<name>/<name>_{1..6}.wav
    <data_root>/concepts/instrument_3/<name>/<name>_{1..3}.wav    (the first 3 segments)

as 44.1 kHz stereo WAV, the format used for training.

Requirements: yt-dlp and ffmpeg on PATH.
Usage:
    python scripts/download_instruments.py --data-root <dataset_root>/data
    python scripts/download_instruments.py --data-root <dataset_root>/data --only violin,cello
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

SR, CHANNELS = 44100, 2


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def fetch_audio(ytid, cache_dir):
    """Download the full audio track once (cached as <cache>/<ytid>.wav)."""
    out = cache_dir / f"{ytid}.wav"
    if not out.exists():
        run(["yt-dlp", "-x", "--audio-format", "wav", "-o", str(cache_dir / f"{ytid}.%(ext)s"),
             f"https://www.youtube.com/watch?v={ytid}"])
    return out


def cut(src, start, dur, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", str(src),
         "-ar", str(SR), "-ac", str(CHANNELS), str(dst)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="<dataset_root>/data (contains concepts/)")
    ap.add_argument("--only", default="", help="comma-separated instrument names to process")
    ap.add_argument("--cache-dir", default=None, help="where full downloads are kept (default: <data-root>/.yt_cache)")
    args = ap.parse_args()

    data = Path(args.data_root)
    meta = data / "concepts" / "metadata_instruments.csv"
    cache = Path(args.cache_dir) if args.cache_dir else data / ".yt_cache"
    cache.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    rows = list(csv.DictReader(open(meta, newline="")))
    failed = []
    for r in rows:
        name = r["concept_name"].strip()
        if only and name not in only:
            continue
        n, dur, start = int(r["num_clips"]), float(r["clip_duration_sec"]), float(r["start_sec"])
        d6 = data / "concepts" / f"instrument_{n}" / name
        d3 = data / "concepts" / "instrument_3" / name
        if all((d6 / f"{name}_{i}.wav").exists() for i in range(1, n + 1)):
            print(f"[skip] {name}")
            continue
        try:
            src = fetch_audio(r["ytid"].strip(), cache)
            for i in range(n):
                cut(src, start + i * dur, dur, d6 / f"{name}_{i + 1}.wav")
            for i in range(1, 4):
                d3.mkdir(parents=True, exist_ok=True)
                shutil.copy(d6 / f"{name}_{i}.wav", d3 / f"{name}_{i}.wav")
            print(f"[ok]   {name}: {n} clips from {r['ytid']} @ {start:.0f}s")
        except subprocess.CalledProcessError as e:
            failed.append(name)
            print(f"[fail] {name}: {e.stderr.decode(errors='ignore').strip()[-200:]}")
    if failed:
        print(f"\n{len(failed)} instrument(s) failed (video removed or region-locked?): {', '.join(failed)}")


if __name__ == "__main__":
    main()
