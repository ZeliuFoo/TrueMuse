"""
DreamBooth fine-tuning for Stable Audio Open (no prior preservation).

Usage:
    python dreambooth_stable_audio.py \
        --instance_data_dir /path/to/audio/files \
        --output_dir ./bagpipe_sa_db \
        --object_class bagpipe \
        --instance_word sks \
        --max_train_steps 300
"""

import argparse
import gc
import os
import shutil
import logging

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from einops import rearrange
from scipy.io.wavfile import write as write_wav

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference.sampling import get_alphas_sigmas

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir",        type=str, required=True)
    parser.add_argument("--object_class",      type=str, required=True)
    parser.add_argument("--instance_word",     type=str, default="sks")
    parser.add_argument("--prompt_prefix",     type=str, default="a recording of")
    parser.add_argument("--max_train_steps",   type=int, default=300)
    parser.add_argument("--learning_rate",     type=float, default=1e-4)
    parser.add_argument("--train_batch_size",  type=int, default=1)
    parser.add_argument("--repeats",           type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--validation_steps",  type=int, default=0,
        help="Generate validation audio every N steps. 0 = disabled.")
    parser.add_argument("--num_validation_audio", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=200)
    parser.add_argument("--cfg_scale",         type=float, default=7.0)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--mixed_precision",   type=str, default="fp16", choices=["no", "fp16"])
    parser.add_argument("--window_seconds",    type=int, default=10)
    args = parser.parse_args()

    args.instance_prompt = f"{args.prompt_prefix} {args.instance_word} {args.object_class}"
    return args


class DreamBoothAudioDataset(Dataset):
    def __init__(self, data_dir, prompt, sample_rate, sample_size, audio_channels, repeats):
        self.files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS
        ]
        assert len(self.files) > 0, f"No audio files found in {data_dir}"
        logger.info(f"Found {len(self.files)} audio files in {data_dir}")

        self.prompt = prompt
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.audio_channels = audio_channels
        self.items = self.files * repeats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx % len(self.files)]
        wav, sr = torchaudio.load(path)

        # resample
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # channels
        if self.audio_channels == 2 and wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif self.audio_channels == 1 and wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        # loop or crop to sample_size
        if wav.shape[1] < self.sample_size:
            repeats = (self.sample_size + wav.shape[1] - 1) // wav.shape[1]
            wav = wav.repeat(1, repeats)[:, :self.sample_size]
        else:
            start = torch.randint(0, wav.shape[1] - self.sample_size + 1, (1,)).item()
            wav = wav[:, start:start + self.sample_size]

        return wav, self.prompt


def collate_fn(batch):
    wavs, prompts = zip(*batch)
    return torch.stack(wavs), list(prompts)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Instance prompt: '{args.instance_prompt}'")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading Stable Audio Open...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)

        # 加在 model.to(device) 之后
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable params: {len(trainable)}")
    logger.info(f"First few: {trainable[:10]}")

    sample_rate    = model_config["sample_rate"]
    sample_size    = sample_rate * args.window_seconds
    audio_channels = model_config.get("audio_channels", 2)

    # freeze pretransform (VAE) and conditioner (T5 text encoder + timing embedder)
    # only train diffusion transformer (model.model)
    if model.pretransform is not None:
        model.pretransform.requires_grad_(False)
        model.pretransform.eval()
    if model.conditioner is not None:
        model.conditioner.requires_grad_(False)
        model.conditioner.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = DreamBoothAudioDataset(
        args.instance_data_dir, args.instance_prompt,
        sample_rate, sample_size, audio_channels, args.repeats
    )
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    # Save a copy of training audio for reference
    train_audio_dir = os.path.join(args.output_dir, "training_audio")
    os.makedirs(train_audio_dir, exist_ok=True)
    for f in dataset.files:
        shutil.copy(f, train_audio_dir)

    # ── Optimizer（Adam，与官方一致）─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    rng = torch.quasirandom.SobolEngine(1, scramble=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting training: {args.max_train_steps} steps")
    global_step = 0
    model.train()
    progress_bar = tqdm(range(args.max_train_steps), desc="Steps")

    data_iter = iter(dataloader)
    while global_step < args.max_train_steps:
        try:
            reals, prompts = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            reals, prompts = next(data_iter)

        reals = reals.to(device, dtype=torch.float32)

        # encode to latents
        with torch.no_grad():
            if model.pretransform is not None:
                latents = model.pretransform.encode(reals)
            else:
                latents = reals

        # get conditioning
        metadata = [{"prompt": p, "seconds_start": 0, "seconds_total": args.window_seconds}
                    for p in prompts]
        conditioning = model.conditioner(metadata, device)

        # sample timesteps
        t = rng.draw(reals.shape[0])[:, 0].to(device)
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]

        noise = torch.randn_like(latents)
        noised = latents * alphas + noise * sigmas
        targets = noise * alphas - latents * sigmas  # v-prediction

        # forward
        if args.mixed_precision == "fp16":
            with torch.amp.autocast("cuda"):
                output = model(noised, t, cond=conditioning, cfg_dropout_prob=0.1)
                loss = torch.nn.functional.mse_loss(output, targets)
        else:
            output = model(noised, t, cond=conditioning, cfg_dropout_prob=0.1)
            loss = torch.nn.functional.mse_loss(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step % args.checkpointing_steps == 0:
            model_dir = os.path.join(args.output_dir, f"model_step_{global_step}")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(model_dir, "pytorch_model.ckpt"),
            )
            logger.info(f"Saved checkpoint: {model_dir}")

        # ── Validation ────────────────────────────────────────────────────────
        if args.validation_steps > 0 and global_step % args.validation_steps == 0:
            logger.info(f"Generating validation audio at step {global_step}...")
            val_dir = os.path.join(args.output_dir, f"val_audio_{global_step}")
            os.makedirs(val_dir, exist_ok=True)
            model.eval()

            for i in range(args.num_validation_audio):
                with torch.no_grad():
                    audio = generate_diffusion_cond(
                        model,
                        steps=args.num_inference_steps,
                        cfg_scale=args.cfg_scale,
                        conditioning=[{
                            "prompt": args.instance_prompt,
                            "seconds_start": 0,
                            "seconds_total": args.window_seconds,
                        }],
                        sample_size=sample_rate * args.window_seconds,
                        device=device,
                        sampler_type="dpmpp-3m-sde",
                        sigma_min=0.3,
                        sigma_max=500,
                    )
                audio = rearrange(audio, "b d n -> d (b n)")
                audio = audio.float().div(torch.max(torch.abs(audio))).clamp(-1, 1)
                audio_np = audio.cpu().numpy().T
                fname = os.path.join(val_dir, f"{args.instance_prompt.replace(' ', '_')}_{i}.wav")
                write_wav(fname, sample_rate, audio_np)
                logger.info(f"  Saved: {fname}")

            model.train()
            if model.pretransform is not None:
                model.pretransform.eval()
            gc.collect()
            torch.cuda.empty_cache()

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "model_final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict()},
        os.path.join(final_dir, "pytorch_model.ckpt"),
    )
    logger.info(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()