"""DAC (Descript Audio Codec) 16 kHz wrapper.

Uses codebook histograms: for each of Nq residual codebooks (each size V),
count how many times each code is used across T time-steps, then concat the
Nq histograms → (Nq*V,)-dim vector, L2-normalize, use cosine similarity.
"""

import numpy as np
import torch
import torchaudio
import dac


class DACEncoder:
    def __init__(self, sample_rate: str = "16khz", device: str = "cuda"):
        self.device = device
        ckpt = dac.utils.download(model_type=sample_rate)
        self.model = dac.DAC.load(ckpt).to(device).eval()
        self.sr = self.model.sample_rate        # 16000
        self.nq = self.model.n_codebooks        # typ. 12 @ 16 kHz
        self.vocab = self.model.codebook_size   # typ. 1024
        self.dim = self.nq * self.vocab

    def _resample(self, wav: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr != self.sr:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), orig_sr, self.sr
            ).numpy()
        return wav

    @torch.no_grad()
    def embed(self, wav: np.ndarray, orig_sr: int) -> np.ndarray:
        return self.embed_batch([wav], orig_sr)[0]

    @torch.no_grad()
    def embed_batch(self, wavs: list, orig_sr: int) -> np.ndarray:
        """wavs: list of 1-D float32 arrays. Returns (B, D) L2-normalized embeddings."""
        wavs_rs = [self._resample(w, orig_sr) for w in wavs]
        x = torch.from_numpy(np.stack(wavs_rs)).float().unsqueeze(1).to(self.device)  # (B, 1, T)
        x = self.model.preprocess(x, self.sr)
        _, codes, _, _, _ = self.model.encode(x)  # codes: (B, Nq, T_codes)
        codes = codes.cpu().numpy()

        hists = []
        for b in range(len(wavs)):
            hist = np.zeros(self.dim, dtype=np.float32)
            for q in range(self.nq):
                counts = np.bincount(codes[b, q], minlength=self.vocab).astype(np.float32)
                hist[q * self.vocab:(q + 1) * self.vocab] = counts
            n = np.linalg.norm(hist)
            if n > 0:
                hist /= n
            hists.append(hist)
        return np.stack(hists)
