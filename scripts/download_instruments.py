"""Rebuild the instrument concept clips from YouTube.

The instrument audio is not redistributed. Its YouTube sources (video ID + start time) are listed in
metadata_instruments.csv next to this script. This script downloads each video's audio once and cuts
`num_clips` consecutive `clip_duration_sec`-second segments starting at `start_sec`, then writes

    <data_root>/concepts/instrument_6/<name>/<name>_{1..6}.wav
    <data_root>/concepts/instrument_3/<name>/<name>_{1..3}.wav    (the first 3 segments)

as 44.1 kHz stereo WAV, the format used for training.

Requirements: ffmpeg and a recent yt-dlp on PATH (pip install -U "yt-dlp[default]"; old versions get HTTP 403).
Usage:
    python scripts/download_instruments.py --data-root <dataset_root>/data
    python scripts/download_instruments.py --data-root <dataset_root>/data --only violin,cello
Re-run the same command to retry instruments that failed; finished instruments are skipped.
"""
import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

SR, CHANNELS = 44100, 2
DEFAULT_META = Path(__file__).with_name("metadata_instruments.csv")


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def fetch_audio(ytid, cache_dir, ffmpeg, attempts=3):
    """Download the full audio track once (cached as <cache>/<ytid>.wav); YouTube errors are often transient."""
    out = cache_dir / f"{ytid}.wav"
    for k in range(attempts):
        if out.exists():
            break
        try:
            # an explicit --ffmpeg-location avoids a LookupError in some yt-dlp versions
            run(["yt-dlp", "-x", "--audio-format", "wav", "--ffmpeg-location", ffmpeg,
                 "-o", str(cache_dir / f"{ytid}.%(ext)s"), f"https://www.youtube.com/watch?v={ytid}"])
        except subprocess.CalledProcessError:
            if k == attempts - 1:
                raise
            time.sleep(10)
    return out


def error_text(e):
    lines = e.stderr.decode(errors="ignore").strip().splitlines()
    errors = [l for l in lines if l.startswith("ERROR")]
    return (errors or lines or ["unknown error"])[-1][:300]


def cut(src, start, dur, dst, ffmpeg):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", str(src),
         "-ar", str(SR), "-ac", str(CHANNELS), str(dst)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="<dataset_root>/data (contains concepts/)")
    ap.add_argument("--metadata", default=str(DEFAULT_META), help="instrument sources (default: next to this script)")
    ap.add_argument("--only", default="", help="comma-separated instrument names to process")
    ap.add_argument("--cache-dir", default=None, help="where full downloads are kept (default: <data-root>/.yt_cache)")
    args = ap.parse_args()

    data = Path(args.data_root)
    meta = Path(args.metadata)
    cache = Path(args.cache_dir) if args.cache_dir else data / ".yt_cache"
    cache.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("yt-dlp"):
        raise SystemExit("yt-dlp and ffmpeg must both be on PATH")

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
            src = fetch_audio(r["ytid"].strip(), cache, ffmpeg)
            for i in range(n):
                cut(src, start + i * dur, dur, d6 / f"{name}_{i + 1}.wav", ffmpeg)
            for i in range(1, 4):
                d3.mkdir(parents=True, exist_ok=True)
                shutil.copy(d6 / f"{name}_{i}.wav", d3 / f"{name}_{i}.wav")
            print(f"[ok]   {name}: {n} clips from {r['ytid']} @ {start:.0f}s")
        except subprocess.CalledProcessError as e:
            failed.append(name)
            print(f"[fail] {name}: {error_text(e)}")
    if failed:
        print(f"\n{len(failed)} instrument(s) failed: {', '.join(failed)}. Re-run the command to retry them.")


if __name__ == "__main__":
    main()
