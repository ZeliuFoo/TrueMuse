# Third-party code

The fine-tuning and generation scripts are adapted from, and run inside, the official code of each generator.
These components keep their original licenses.

| Path in this repo | Derived from | License |
|---|---|---|
| `finetuning/mustango/models.py`, `mustango.py`, `tango.py`, `modelling_deberta_v2.py`, `dreambooth_mustango.py` | [AMAAI-Lab/mustango](https://github.com/AMAAI-Lab/mustango) (Copyright (c) 2023 DeCLaRe Lab); `modelling_deberta_v2.py` is based on Hugging Face Transformers | MIT; Transformers code under Apache-2.0 |
| `finetuning/stable_audio/dreambooth_stable_audio.py`, `finetuning/stable_audio/setup.py` | [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) (Copyright (c) Stability AI) | MIT |
| `finetuning/audioldm2/dreambooth_audioldm2.py` | [zelaki/DreamSound](https://github.com/zelaki/DreamSound) (Plitsis et al., "Investigating Personalization Methods in Text to Music Generation") and the Hugging Face Diffusers DreamBooth example | see the upstream repositories |

The AudioLDM2 scripts import modules (`pipeline/`, `audioldm/`, `utils/`, `evaluate.py`) that are **not** included
here; obtain them from the DreamSound repository.

Model weights are downloaded from their official Hugging Face repositories and are subject to their own
licenses: `cvssp/audioldm2`, `declare-lab/mustango`, `stabilityai/stable-audio-open-1.0` (Stability AI
Community License; gated), `m-a-p/MERT-v0`, `m-a-p/MERT-v0-public`, `m-a-p/music2vec-v1`, and Descript Audio Codec.
