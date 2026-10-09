"""
DreamBooth fine-tuning for Mustango (MusicAudioDiffusion).

Key differences from DreamSound (AudioLDM2):
- Mustango UNet takes beats + chords as extra conditioning (beyond text)
- Beats/chords are predicted from the text prompt via MusicFeaturePredictor
- Trainable: unet + beat_embedding_layer + chord_embedding_layer
- Frozen:    text_encoder, vae, stft, FME (Fundamental_Music_Embedding)

Usage:
    python dreambooth_mustango.py \
        --pretrained_model_path /path/to/mustango/checkpoint \
        --instance_data_dir    /path/to/instrument/audio_files \
        --output_dir           /path/to/output \
        --instance_word        sks \
        --object_class         didgeridoo \

        --max_train_steps      300
"""

import argparse
import gc
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.io.wavfile import write as write_wav

import tools.torch_tools as torch_tools
from models import MusicAudioDiffusion
from mustango import MusicFeaturePredictor

logger = get_logger(__name__)

# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--pretrained_model_path", type=str,
        default="declare-lab/mustango",
        help="Path to local Mustango checkpoint directory")
    parser.add_argument("--text_encoder_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--scheduler_name",    type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--unet_model_config", type=str, default=None,
        help="Path to munet config json. Defaults to <pretrained_model_path>/configs/music_diffusion_model_config.json")

    # Data
    parser.add_argument("--instance_data_dir", type=str, required=True,
        help="Directory containing instance audio files (.wav / .mp3)")
    parser.add_argument("--class_data_dir", type=str, default=None,
        help="Directory for class (prior) audio files. Auto-created if --with_prior_preservation.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_word",  type=str, default="sks",
        help="Rare token used as instance identifier")
    parser.add_argument("--object_class",   type=str, required=True,
        help="Generic class name, e.g. 'didgeridoo'")
    parser.add_argument("--instance_prompt_prefix", type=str, default="a recording of",
        help="Prefix for instance prompts: '<prefix> <instance_word> <object_class>'")
    parser.add_argument("--class_prompt_prefix", type=str, default="a recording of",
        help="Prefix for class prompts: '<prefix> <object_class>'")
    parser.add_argument("--repeats", type=int, default=100,
        help="How many times to repeat the instance dataset per epoch")

    # Prior preservation
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)

    # Training
    parser.add_argument("--max_train_steps",     type=int,   default=300)
    parser.add_argument("--learning_rate",       type=float, default=1e-6)
    parser.add_argument("--train_text_encoder",  action="store_true",
        help="Also fine-tune the T5 text encoder so sks embedding can adapt")
    parser.add_argument("--freeze_music_encoder", action="store_true",
        help="Freeze beat/chord embedding layers, only train UNet")
    parser.add_argument("--train_batch_size",    type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpointing_steps", type=int,   default=150)
    parser.add_argument("--snr_gamma",           type=float, default=5.0,
        help="SNR rebalancing weight (set None to disable)")
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--mixed_precision",     type=str,   default="fp16",
        choices=["no", "fp16", "bf16"])

    # Inference (validation)
    parser.add_argument("--num_validation_audio", type=int, default=2)
    parser.add_argument("--validation_steps",     type=int, default=0,
        help="Generate validation audio every N steps. 0 = disabled.")
    parser.add_argument("--num_inference_steps",  type=int, default=100)
    parser.add_argument("--guidance_scale",        type=float, default=3.0)

    args = parser.parse_args()

    if args.unet_model_config is None:
        args.unet_model_config = os.path.join(
            args.pretrained_model_path, "configs", "music_diffusion_model_config.json"
        )
    if args.class_data_dir is None:
        args.class_data_dir = os.path.join(args.output_dir, "class_audio")

    args.instance_prompt = f"{args.instance_prompt_prefix} {args.instance_word} {args.object_class}"
    args.class_prompt    = f"{args.class_prompt_prefix} {args.object_class}"

    return args


# ── Dataset ────────────────────────────────────────────────────────────────────

class DreamBoothAudioDataset(Dataset):
    """
    Loads instance audio files (and optionally class audio files for prior preservation).
    Beats and chords are predicted from the prompt text via MusicFeaturePredictor,
    consistent with how Mustango does inference.
    """

    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

    def __init__(
        self,
        instance_data_dir,
        instance_prompt,
        music_feature_predictor,
        class_data_dir=None,
        class_prompt=None,
        with_prior_preservation=False,
        repeats=100,
        audio_length_s=10,
        sample_rate=16000,
    ):
        self.instance_prompt = instance_prompt
        self.class_prompt    = class_prompt
        self.with_prior      = with_prior_preservation
        self.repeats         = repeats
        self.target_length   = int(audio_length_s * sample_rate)
        self.sample_rate     = sample_rate
        self.mfp             = music_feature_predictor

        self.instance_paths = sorted([
            str(p) for p in Path(instance_data_dir).iterdir()
            if p.suffix.lower() in self.AUDIO_EXTS
        ])
        assert len(self.instance_paths) > 0, f"No audio files found in {instance_data_dir}"

        if with_prior_preservation:
            assert class_data_dir and class_prompt, "Need class_data_dir and class_prompt for prior preservation"
            self.class_paths = sorted([
                str(p) for p in Path(class_data_dir).iterdir()
                if p.suffix.lower() in self.AUDIO_EXTS
            ])
            assert len(self.class_paths) > 0, f"No class audio files found in {class_data_dir}"

        # Pre-compute beat/chord features for each unique prompt (cheap, text-based)
        self._instance_beats, self._instance_chords, self._instance_chords_time = \
            self._get_music_features(instance_prompt)
        if with_prior_preservation:
            self._class_beats, self._class_chords, self._class_chords_time = \
                self._get_music_features(class_prompt)

    def _get_music_features(self, prompt):
        predicted_beats, chords, chords_time = self.mfp.generate(prompt)
        # mfp returns [[times, counts]] (batch-wrapped); unwrap to [times, counts] per sample
        # empty case: [[], []] → keep as [[], []]
        if not predicted_beats or predicted_beats == [[], []]:
            beat = [[], []]
        else:
            beat = predicted_beats[0]  # [times, counts]
        return beat, chords, chords_time

    def __len__(self):
        return len(self.instance_paths) * self.repeats

    def __getitem__(self, idx):
        instance_path = self.instance_paths[idx % len(self.instance_paths)]
        item = {
            "instance_audio":       instance_path,   # pass path; wav_to_fbank loads it
            "instance_prompt":      self.instance_prompt,
            "instance_beats":       self._instance_beats,
            "instance_chords":      self._instance_chords,
            "instance_chords_time": self._instance_chords_time,
        }
        if self.with_prior:
            class_path = self.class_paths[idx % len(self.class_paths)]
            item.update({
                "class_audio":       self._load_audio(class_path),
                "class_prompt":      self.class_prompt,
                "class_beats":       self._class_beats,
                "class_chords":      self._class_chords,
                "class_chords_time": self._class_chords_time,
            })
        return item


def collate_fn(examples):
    """Collect paths and features; wav_to_fbank handles audio loading."""
    batch = {
        "instance_audio":       [e["instance_audio"] for e in examples],  # list of paths
        "instance_prompt":      [e["instance_prompt"] for e in examples],
        "instance_beats":       [e["instance_beats"] for e in examples],
        "instance_chords":      [e["instance_chords"] for e in examples],
        "instance_chords_time": [e["instance_chords_time"] for e in examples],
    }
    if "class_audio" in examples[0]:
        batch.update({
            "class_audio":       [e["class_audio"] for e in examples],  # list of paths
            "class_prompt":      [e["class_prompt"] for e in examples],
            "class_beats":       [e["class_beats"] for e in examples],
            "class_chords":      [e["class_chords"] for e in examples],
            "class_chords_time": [e["class_chords_time"] for e in examples],
        })
    return batch


# ── Audio → latent helper ──────────────────────────────────────────────────────

def audio_to_latent(paths, vae, stft, target_length=1024):
    """
    paths: list of audio file paths
    Returns: scaled latents via audioldm VAE (scale_factor * z)
    Uses wav_to_fbank which applies the correct transpose + padding,
    matching exactly what train.py does.
    """
    device = next(vae.parameters()).device
    fbank, _, _ = torch_tools.wav_to_fbank(paths, target_length, stft)
    fbank = fbank.unsqueeze(1).to(device, dtype=torch.float32)  # (B, 1, T, n_mel)
    with torch.no_grad():
        posterior = vae.encode_first_stage(fbank)
        latents = vae.get_first_stage_encoding(posterior)       # scale_factor * z
    return latents


# ── Class audio generation (prior preservation) ────────────────────────────────

@torch.no_grad()
def generate_class_audio(model, vae, stft, class_prompt, beats, chords, chords_time,
                          num_samples, output_dir, num_steps=20, guidance_scale=3.0):
    """Generate class audio samples with the pretrained model before training."""
    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    if len(existing) >= num_samples:
        logger.info(f"  Found {len(existing)} class audio files, skipping generation.")
        return

    logger.info(f"Generating {num_samples} class audio samples for prior preservation...")
    device = next(model.parameters()).device
    model.eval()

    from diffusers import DDPMScheduler
    inference_scheduler = DDPMScheduler.from_pretrained(
        model.scheduler_name, subfolder="scheduler"
    )

    generated = len(existing)
    pbar = tqdm(total=num_samples - generated, desc="Generating class audio")

    while generated < num_samples:
        latents = model.inference(
            [class_prompt], [beats], [chords], [chords_time],
            inference_scheduler,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            num_samples_per_prompt=1,
            disable_progress=True,
        )
        # decode_first_stage handles 1/scale_factor internally
        mel = vae.decode_first_stage(latents)          # (1, 1, mel, T)
        wav = vae.decode_to_waveform(mel)              # (1, T) numpy via vocoder
        out_path = os.path.join(output_dir, f"class_{generated:04d}.wav")
        write_wav(out_path, 16000, wav[0])

        generated += 1
        pbar.update(1)

    pbar.close()
    model.train()


# ── Training ───────────────────────────────────────────────────────────────────

def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod     = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    snr = (alpha / sigma) ** 2
    return snr


def training_step(model, vae, stft, batch, noise_scheduler, snr_gamma,
                  with_prior, prior_loss_weight):
    device = next(model.parameters()).device

    # ── Instance latents ──────────────────────────────────────────────────────
    inst_latents = audio_to_latent(batch["instance_audio"], vae, stft)
    bsz = inst_latents.shape[0]

    if with_prior:
        cls_latents = audio_to_latent(batch["class_audio"].to(device), vae, stft)
        # Concatenate: first half = instance, second half = class
        latents = torch.cat([inst_latents, cls_latents], dim=0)
        prompts = batch["instance_prompt"] + batch["class_prompt"]
        beats   = batch["instance_beats"]  + batch["class_beats"]
        chords  = batch["instance_chords"] + batch["class_chords"]
        chords_time = batch["instance_chords_time"] + batch["class_chords_time"]
    else:
        latents     = inst_latents
        prompts     = batch["instance_prompt"]
        beats       = batch["instance_beats"]
        chords      = batch["instance_chords"]
        chords_time = batch["instance_chords_time"]

    # ── Noise ─────────────────────────────────────────────────────────────────
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Target
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    encoder_hidden_states, boolean_encoder_mask = model.encode_text(prompts)
    encoded_beats, beat_mask   = model.encode_beats(beats)
    encoded_chords, chord_mask = model.encode_chords(chords, chords_time)

    model_pred = model.unet(
        noisy_latents, timesteps, encoder_hidden_states,
        encoded_beats, encoded_chords,
        encoder_attention_mask=boolean_encoder_mask,
        beat_attention_mask=beat_mask,
        chord_attention_mask=chord_mask,
    ).sample

    # ── Loss ──────────────────────────────────────────────────────────────────
    if snr_gamma is not None:
        snr = compute_snr(noise_scheduler, timesteps)
        mse_weights = (
            torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        raw_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        raw_loss = raw_loss.mean(dim=list(range(1, len(raw_loss.shape)))) * mse_weights
    else:
        raw_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        raw_loss = raw_loss.mean(dim=list(range(1, len(raw_loss.shape))))

    if with_prior:
        instance_loss = raw_loss[:bsz].mean()
        prior_loss    = raw_loss[bsz:].mean()
        loss = instance_loss + prior_loss_weight * prior_loss
    else:
        loss = raw_loss.mean()

    return loss


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Save class_name.json (mirrors DreamSound) ─────────────────────────────
    import json as _json
    with open(os.path.join(args.output_dir, "class_name.json"), "w") as f:
        _json.dump({"object_class": args.object_class, "instance_word": args.instance_word}, f)

    # ── Copy training audio ───────────────────────────────────────────────────
    import shutil
    train_audio_dir = os.path.join(args.output_dir, "training_audio")
    os.makedirs(train_audio_dir, exist_ok=True)
    for p in Path(args.instance_data_dir).iterdir():
        if p.suffix.lower() in {".wav", ".mp3", ".flac"}:
            shutil.copy(str(p), train_audio_dir)

    # ── Load models (mirrors Mustango.__init__) ───────────────────────────────
    logger.info("Loading pretrained models...")
    path = args.pretrained_model_path  # snapshot_download path

    vae_config  = json.load(open(f"{path}/configs/vae_config.json"))
    stft_config = json.load(open(f"{path}/configs/stft_config.json"))
    main_config = json.load(open(f"{path}/configs/main_config.json"))

    from audioldm.audio.stft import TacotronSTFT
    from audioldm.variational_autoencoder import AutoencoderKL
    vae  = AutoencoderKL(**vae_config)
    stft = TacotronSTFT(**stft_config)

    vae.load_state_dict(torch.load(f"{path}/vae/pytorch_model_vae.bin",   map_location="cpu"))
    stft.load_state_dict(torch.load(f"{path}/stft/pytorch_model_stft.bin", map_location="cpu"))

    model = MusicAudioDiffusion(
        text_encoder_name=main_config["text_encoder_name"],
        scheduler_name=main_config["scheduler_name"],
        unet_model_config_path=f"{path}/configs/music_diffusion_model_config.json",
        snr_gamma=args.snr_gamma,
        freeze_text_encoder=True,
        uncondition=True,  # enables CFG training (10% null conditioning), matches official train.py
    )
    model.load_state_dict(
        torch.load(f"{path}/ldm/pytorch_model_ldm.bin", map_location="cpu")
    )
    logger.info(f"Loaded checkpoint from {path}")

    # Freeze everything, then unfreeze trainable parts
    model.requires_grad_(False)
    vae.requires_grad_(False)
    stft.requires_grad_(False)

    model.unet.requires_grad_(True)
    if not args.freeze_music_encoder:
        model.beat_embedding_layer.requires_grad_(True)
        model.chord_embedding_layer.requires_grad_(True)
    if args.train_text_encoder:
        model.text_encoder.requires_grad_(True)

    vae.eval()
    stft.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ── MusicFeaturePredictor ─────────────────────────────────────────────────
    logger.info("Loading MusicFeaturePredictor...")
    mfp = MusicFeaturePredictor(
        path=args.pretrained_model_path, device=accelerator.device
    )

    # ── Prior: generate class audio ───────────────────────────────────────────
    if False:
        beats, chords, chords_time = mfp.generate(args.class_prompt)
        generate_class_audio(
            model, vae, stft,
            class_prompt=args.class_prompt,
            beats=beats, chords=chords, chords_time=chords_time,
            num_samples=args.num_class_audio,
            output_dir=args.class_data_dir,
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = DreamBoothAudioDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        music_feature_predictor=mfp,
        class_data_dir=args.class_data_dir if False else None,
        class_prompt=args.class_prompt if False else None,
        with_prior_preservation=False,
        repeats=args.repeats,
    )

    if args.validation_steps > 0:
        # Pre-compute validation beats/chords (same prompt, no need to recompute each time)
        val_beats, val_chords, val_chords_time = mfp.generate(args.instance_prompt)

    torch.cuda.empty_cache()

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    trainable_params = (
        list(model.unet.parameters()) +
        ([] if args.freeze_music_encoder else
            list(model.beat_embedding_layer.parameters()) +
            list(model.chord_embedding_layer.parameters())) +
        (list(model.text_encoder.parameters()) if args.train_text_encoder else [])
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    num_update_steps = math.ceil(args.max_train_steps / args.gradient_accumulation_steps)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # ── Accelerate prepare ────────────────────────────────────────────────────
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    vae  = vae.to(accelerator.device)
    stft = stft.to(accelerator.device)

    noise_scheduler = model.module.noise_scheduler if hasattr(model, "module") else model.noise_scheduler

    if args.validation_steps > 0:
        # Separate scheduler for validation inference (don't corrupt training scheduler state)
        from diffusers import DDPMScheduler
        inference_scheduler = DDPMScheduler.from_pretrained(
            args.scheduler_name, subfolder="scheduler"
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting training: {args.max_train_steps} steps")
    logger.info(f"Instance prompt: '{args.instance_prompt}'")
    logger.info(f"Class prompt:    '{args.class_prompt}'")

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)
    model.train()

    for epoch in range(math.ceil(args.max_train_steps / len(dataloader))):
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = training_step(
                    model=model.module if hasattr(model, "module") else model,
                    vae=vae, stft=stft,
                    batch=batch,
                    noise_scheduler=noise_scheduler,
                    snr_gamma=args.snr_gamma,
                    with_prior=False,
                    prior_loss_weight=args.prior_loss_weight,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(loss=loss.item())

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)

                        model_dir = os.path.join(args.output_dir, f"model_step_{global_step}")
                        os.makedirs(model_dir, exist_ok=True)
                        torch.save(unwrapped.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
                        shutil.copytree(
                            os.path.join(args.pretrained_model_path, "configs"),
                            os.path.join(model_dir, "configs"),
                            dirs_exist_ok=True,
                        )
                        logger.info(f"Saved model at step {global_step}: {model_dir}")
                        gc.collect()
                        torch.cuda.empty_cache()

                # Validation audio
                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        val_dir = os.path.join(args.output_dir, f"val_audio_{global_step}")
                        os.makedirs(val_dir, exist_ok=True)
                        logger.info(f"Generating validation audio at step {global_step}...")
                        unwrapped.eval()
                        with torch.no_grad():
                            val_latents = unwrapped.inference(
                                [args.instance_prompt],
                                val_beats, [val_chords], [val_chords_time],
                                inference_scheduler,
                                num_steps=args.num_inference_steps,
                                guidance_scale=args.guidance_scale,
                                num_samples_per_prompt=1,
                                disable_progress=True,
                            )
                            val_mel = vae.decode_first_stage(val_latents)
                            val_wav = vae.decode_to_waveform(val_mel)
                        fname = args.instance_prompt.replace(" ", "_") + "_0.wav"
                        write_wav(os.path.join(val_dir, fname), 16000, val_wav[0])
                        logger.info(f"Validation audio saved: {val_dir}/{fname}")
                        unwrapped.train()
                        gc.collect()
                        torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # ── Final save ────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        final_dir = os.path.join(args.output_dir, f"model_step_{global_step}")
        os.makedirs(final_dir, exist_ok=True)
        torch.save(unwrapped.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
        shutil.copytree(
            os.path.join(args.pretrained_model_path, "configs"),
            os.path.join(final_dir, "configs"),
            dirs_exist_ok=True,
        )
        logger.info(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
