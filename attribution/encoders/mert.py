"""MERT encoder wrapper (MERT-v0 / MERT-v0-public).

Mean-over-time per layer, concat across all (L+1) hidden states, L2-normalize.
Output dim = (L+1) * D  (e.g. 13 * 768 = 9984 for MERT-v0).
"""

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000


class MERTEncoder:
    def __init__(self, model_id: str = "m-a-p/MERT-v0", device: str = "cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        self.sr = self.fe.sampling_rate  # 16000

    def _resample(self, wav: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr != self.sr:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), orig_sr, self.sr
            ).numpy()
        return wav

    @torch.no_grad()
    def embed(self, wav: np.ndarray, orig_sr: int) -> np.ndarray:
        """wav: 1-D float32 array. Returns (D,) L2-normalized embedding."""
        return self.embed_batch([wav], orig_sr)[0]

    @torch.no_grad()
    def embed_batch(self, wavs: list, orig_sr: int) -> np.ndarray:
        """wavs: list of 1-D float32 arrays. Returns (B, D) L2-normalized embeddings."""
        wavs = [self._resample(w, orig_sr) for w in wavs]
        inputs = self.fe(wavs, sampling_rate=self.sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (L+1) tensors, each (B, T, D)
        hs = torch.stack(outputs.hidden_states, dim=0)  # (L+1, B, T, D)
        emb = hs.mean(dim=2).permute(1, 0, 2).reshape(len(wavs), -1)  # (B, (L+1)*D)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb.cpu().float().numpy()
