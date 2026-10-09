# TrueMuse: A Benchmark for Data Attribution in Text-to-Music Models

**Jiawei Yu, Jian Liu**

[**Paper (arXiv)**](https://arxiv.org/abs/2610.00835) | [**Dataset (Hugging Face)**](https://huggingface.co/datasets/AnonymousAuthorsssss/TrueMuse)

TrueMuse is a benchmark for **training-data attribution in text-to-music generation**. Ground truth in large
pretrained models is entangled and cannot be verified, so TrueMuse injects a *known* source set into a
generator through controlled DreamBooth fine-tuning: each fine-tuned model sees exactly one attribute
(a musician, an instrument, a genre or a melody) through a small set of source clips. Attribution methods are
then asked to recover those source clips from the model's generations, among a large pool of distractors.
TrueMuse is therefore a *controlled proxy* with attribute-level ground truth (known source exposure).

| | |
|---|---|
| Attribute types | musician, instrument, genre, melody |
| Attributes | 133 (50 musicians, 25 instruments, 8 genres, 50 melodies) |
| Fine-tuning set sizes | 3 or 6 clips (musician, instrument, genre); 1 clip (melody); 10 s each |
| Generators | AudioLDM2, Mustango, Stable Audio Open |
| Fine-tuned models | 648 |
| Generated clips | 95,456 (4 waveforms per prompt; contextual and style prompts) |
| Distractors | 228,598 ten-second Jamendo segments (JamendoMaxCaps) |
| Encoders evaluated | MERT-v0, MERT-v0-public, Music2Vec-v1, DAC-16k (+ a trained contrastive mapper) |

---

## Repository structure

```
TrueMuse/
├── finetuning/        # DreamBooth fine-tuning, one script per generator
│   ├── audioldm2/     #   run inside the DreamSound repo (see Installation)
│   ├── mustango/      #   run inside the Mustango repo
│   └── stable_audio/  #   run inside stable-audio-tools
├── generation/        # generation scripts, one per generator x task configuration
│   ├── audioldm2/
│   ├── mustango/
│   └── stable_audio/
├── attribution/       # embedding extraction, mapper training, evaluation
│   └── encoders/      #   MERT / Music2Vec / DAC wrappers
└── scripts/
    └── download_instruments.py   # rebuild the instrument source clips from YouTube
```

---

## 1. Data

### Download

The data is hosted on Hugging Face as 8 zip archives (about 150 GB in total). You only need the generated
audio of the generators you evaluate; the Jamendo embeddings archive is large (about 39 GB) but is
required for evaluation.

```bash
hf download AnonymousAuthorsssss/TrueMuse --repo-type dataset --local-dir <dataset_root>
cd <dataset_root>
unzip -q truemuse_concepts.zip -d data                                                # no data/ prefix
unzip -q truemuse_generated_stable_audio_musician.zip -d data/generated/stable-audio    # no data/ prefix
for f in truemuse_generated_audioldm2.zip truemuse_generated_mustango.zip \
         truemuse_generated_stable_audio_{genre,instrument,melody}.zip truemuse_embeddings_jamendo.zip; do
    unzip -q "$f"
done
```

### Instrument source clips

The instrument source audio is **not redistributed**. [`scripts/metadata_instruments.csv`](scripts/metadata_instruments.csv)
lists the YouTube ID and start time of each instrument; the script below downloads the audio and cuts 6 consecutive
10-second clips (the first 3 form the 3-clip set). It needs `ffmpeg` and a recent `yt-dlp` on `PATH`
(`pip install -U "yt-dlp[default]"`; older versions get HTTP 403 from YouTube). Instruments whose video can no
longer be downloaded are reported as failed.

```bash
python scripts/download_instruments.py --data-root <dataset_root>/data
```

### Layout expected by the code

```
<dataset_root>/
├── data/
│   ├── concepts/                    # source (fine-tuning) clips per attribute
│   │   ├── musician_3/  musician_6/     composer_XXX/*.wav
│   │   ├── instrument_3/ instrument_6/  <instrument>/<instrument>_{1..6}.wav  (from the script above)
│   │   ├── genre_3/     genre_6/        <genre>/clip_*.wav
│   │   ├── melody_1/                    melody_XXX/*.wav
│   │   └── metadata_instruments.csv
│   ├── generated/                   # generated queries
│   │   └── {audioldm2, mustango, stable-audio}/<task>/<concept>_{ldm2,mustango,sa}_db/{contextual,style}/*.wav
│   └── embeddings/jamendo/          # distractor embeddings: {mert-v0, mert-v0-public, music2vec-v1, dac-16k}/pool.npz
└── embeddings/                      # written by attribution/extract_embeddings_v2.py
```

Sources: musician, genre and melody clips come from the Free Music Archive (FMA); musician clips are
instrumental stems (vocals removed with Demucs). Instrument clips come from YouTube.

---

## 2. Installation

**Attribution code** (embedding extraction, mapper, evaluation). Python 3.10, CUDA 11.8:

```bash
conda create -n truemuse python=3.10 && conda activate truemuse
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
pip install transformers==4.30.2 accelerate==0.23.0
pip install librosa==0.9.2 soundfile==0.12.1 descript-audio-codec==1.0.0
pip install numpy==1.26.4 scipy==1.11.4 tqdm "setuptools<81"    # librosa 0.9.2 needs pkg_resources
```

**Generators.** Fine-tuning and generation need a CUDA GPU and build on each generator's official code. Use one environment per
generator, clone the upstream repository, and copy our scripts into its root:

| Generator | Upstream code | Base checkpoint | Copy into the upstream root | Environment |
|---|---|---|---|---|
| AudioLDM2 | [zelaki/DreamSound](https://github.com/zelaki/DreamSound) | `cvssp/audioldm2` | `finetuning/audioldm2/*.py`, `generation/audioldm2/*.py` | `finetuning/audioldm2/requirements.txt` |
| Mustango | [AMAAI-Lab/mustango](https://github.com/AMAAI-Lab/mustango) | `declare-lab/mustango` | `finetuning/mustango/*.py` (replaces the upstream `models.py`, `mustango.py`, `tango.py`), `generation/mustango/*.py` | `finetuning/mustango/requirements.txt`, plus the upstream `diffusers` fork (`pip install -e diffusers`) |
| Stable Audio Open | [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | `stabilityai/stable-audio-open-1.0` (gated: accept the license on Hugging Face and log in) | `finetuning/stable_audio/dreambooth_stable_audio.py`, `generation/stable_audio/*.py` | `git checkout 50049e3` (the version we used; later versions changed the package layout), `pip install -e .`, then `finetuning/stable_audio/requirements.txt` |

---

## 3. Configuration

Paths are set as constants at the top of each script. Update them before running:

| Script | Constant | Set to |
|---|---|---|
| `attribution/extract_embeddings_v2.py` | `DATA` | `<dataset_root>/data` |
| `attribution/extract_embeddings_v2.py` | `OUT_ROOT` | `<dataset_root>/embeddings` |
| `attribution/dataset.py` | `EMB_ROOT` | `<dataset_root>/embeddings` |
| `attribution/train.py` | `JAMENDO_ROOT` | `<dataset_root>/data/embeddings/jamendo` |
| `attribution/evaluate.py`, `attribution/evaluate_pretrained.py` | `JAMENDO_ROOT` | `<dataset_root>/data/embeddings/jamendo` |
| `attribution/evaluate.py`, `attribution/evaluate_pretrained.py` | `OUT_ROOT` | directory for result CSVs |
| `generation/*/generate_all_*.py` | `BASE` | root that holds the fine-tuned models (`MODEL_DIR`) and outputs (`OUT_BASE`) |
| `generation/mustango/generate_all_*.py` | `PRETRAINED_PATH` | local snapshot of `declare-lab/mustango` (`hf download declare-lab/mustango`) |

---

## 4. Fine-tuning (DreamBooth)

Every attribute is fine-tuned independently with the prompt `a recording of sks <class>`, where `<class>` is
`musician`, `genre`, `melody`, or the instrument name. All generators share the same recipe:

| Setting | Clips | Steps | Batch size | Learning rate |
|---|---|---|---|---|
| melody | 1 | 300 | 1 | 4e-6 |
| musician / instrument / genre, 3-clip | 3 | 1000 | 3 | 4e-6 |
| musician / instrument / genre, 6-clip | 6 | 1500 | 4 | 4e-6 |

Name each output directory `<BASE>/<model_dir>/<concept>_<suffix>`, which is where the generation scripts
look for it. `<model_dir>` is one of `composer_models_{3,6}`, `instrument_model_{3,6}`, `genre_models_{3,6}`,
`melody_models_1`; `<suffix>` is `ldm2_db`, `mustango_db` or `sa_db`.

```bash
# AudioLDM2 (inside DreamSound)
accelerate launch dreambooth_audioldm2.py \
    --pretrained_model_name_or_path cvssp/audioldm2 \
    --train_data_dir <dataset_root>/data/concepts/musician_3/composer_001 \
    --instance_word sks --object_class musician \
    --train_batch_size 3 --gradient_accumulation_steps 1 \
    --max_train_steps 1000 --learning_rate 4e-6 \
    --num_vectors 1 --checkpointing_steps 100 --validation_steps 500 \
    --output_dir <BASE>/composer_models_3/composer_001_ldm2_db

# Mustango (inside mustango)
accelerate launch dreambooth_mustango.py \
    --pretrained_model_path <mustango_snapshot> \
    --instance_data_dir <dataset_root>/data/concepts/musician_3/composer_001 \
    --instance_word sks --object_class musician \
    --train_batch_size 3 --max_train_steps 1000 --learning_rate 4e-6 \
    --freeze_music_encoder --checkpointing_steps 100 \
    --output_dir <BASE>/composer_models_3/composer_001_mustango_db

# Stable Audio Open (inside stable-audio-tools)
python dreambooth_stable_audio.py \
    --instance_data_dir <dataset_root>/data/concepts/musician_3/composer_001 \
    --instance_word sks --object_class musician \
    --train_batch_size 3 --max_train_steps 1000 --learning_rate 4e-6 \
    --checkpointing_steps 100 --cfg_scale 3.0 --window_seconds 10 --mixed_precision fp16 \
    --output_dir <BASE>/composer_models_3/composer_001_sa_db
```

For instruments, pass the instrument name as `--object_class` (for example `--object_class violin`).
We fine-tuned on a single A100. Mustango at batch size 3 does not fit on a 24 GB GPU; there, use a smaller
`--train_batch_size` with `--gradient_accumulation_steps` to keep the same effective batch.
AudioLDM2 saves the pipeline that the generation scripts load (`pipeline_step_<N>`) only at validation steps, so
`--validation_steps` must divide the number of training steps: 500 for 1000 or 1500 steps, 300 for melody.

---

## 5. Generation

Pre-generated audio is included in the dataset; these scripts reproduce it. Each script loops over all
concepts of one task configuration and writes `metadata.csv` files with the prompt of every clip.

```bash
python generate_all_musicians_3.py     # also: _musicians_6, _instruments_{3,6}, _genres_{3,6}, _melodies_1
```

| | AudioLDM2 | Mustango | Stable Audio Open |
|---|---|---|---|
| Checkpoint | step 1000 (melody: 300) | step 1000 (melody: 300) | step 1000 (melody: 300) |
| Sampling steps | 200 | 100 | 200 (`dpmpp-3m-sde`, sigma 0.3 to 500) |
| Guidance scale | 3.5 (pipeline default) | 3.0 | 4.0 |
| Waveforms per prompt | 4 | 4 | 4 |
| Output | 10 s, 16 kHz mono | 10.24 s, 16 kHz mono | 10 s, 44.1 kHz stereo |
| Prompts per concept | 15 contextual + 28 style | 15 contextual + 28 style | 15 contextual + 8 style (genre: 28 style) |

*Contextual* prompts place the attribute in a richer musical scene; *style* prompts ask for the attribute in a
given genre. The prompt lists are defined at the top of each generation script.

Outputs are written to `<BASE>/generated_samples_<task>/`. To evaluate your own generations, place them at
`<dataset_root>/data/generated/<generator>/<task>/` (for example `generated_samples_musicians_3` becomes
`data/generated/audioldm2/musician_3`).

---

## 6. Embedding extraction

```bash
cd attribution
python extract_embeddings_v2.py --encoder mert-v0-public --generator all --task all
```

Encoders: `mert-v0`, `mert-v0-public`, `music2vec-v1`, `dac-16k`. Generators: `audioldm2`, `mustango`,
`stable-audio`. Tasks: `musician_{3,6}`, `instrument_{3,6}`, `genre_{3,6}`, `melody_1`, or `all`. Embeddings are
written to `<dataset_root>/embeddings/<encoder>/<generator>/<task>/<concept>/{exemplars,queries}.npz`. The Jamendo
distractor embeddings are already provided in `data/embeddings/jamendo/`
(`extract_jamendo_embeddings.py` recreates them from JamendoMaxCaps audio).

---

## 7. Attribution mapper (optional)

A residual linear mapper (separate exemplar and query branches, zero-initialised) trained with a symmetric
InfoNCE loss on top of a frozen encoder. **The settings used in the paper differ from the script defaults:**

```bash
cd attribution
python train.py --encoder mert-v0-public --generator audioldm2 --tasks musician_3 --prompt both \
    --lr 4e-6 --epochs 500 --patience 100 --batch-size 32 --lam-reg 0.05 --out-dir checkpoints
```

`--prompt` is `contextual`, `style` or `both`; `--tasks` accepts a comma-separated list for joint training
(for example `musician_3,instrument_3,genre_3,melody_1`).

---

## 8. Evaluation

```bash
cd attribution
python evaluate_pretrained.py                                   # frozen encoders, cosine similarity
python evaluate.py --ckpt-dir checkpoints --out-dir results      # trained mappers
python evaluate.py --ckpt-dir checkpoints --out-dir results --test-generator all   # cross-generator
```

**Protocol.** Attributes are split 70 / 15 / 15 into train / validation / test (seed 42); the split is shared
by all encoders and generators. For every test query, the gallery contains the source clips of all test
attributes of the same task (for example `musician_3`) plus the 228,598 Jamendo distractors. **R@k** is the fraction of the query's own
source clips that appear in the top k; we also report mAP. Results are reported per generator: absolute
scores depend on how closely a generator reproduces its fine-tuning clips, so methods should be compared
within a generator.

---

## License

- **Code:** MIT (see [LICENSE](LICENSE)). The fine-tuning and generation scripts build on third-party code;
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Data:** see the [dataset card](https://huggingface.co/datasets/AnonymousAuthorsssss/TrueMuse). FMA-derived
  clips keep the Creative Commons license of their source tracks (listed per clip in `concept_licenses.csv`
  on Hugging Face); instrument audio is not redistributed.

## Intended use

TrueMuse is released for research on data attribution and credit assignment in music generation. It must not
be used to imitate or impersonate specific artists, for commercial style cloning, to train models that
reproduce individual artists, or to redistribute audio that closely reproduces a source recording. Access to
the benchmark grants no rights to any underlying recording, composition, performance or artist identity.

**Takedown and opt-out:** rights holders who want an attribute removed from the benchmark can open an issue on
this repository.

## Citation

If you use TrueMuse, please cite:

```bibtex
@article{yu2026truemuse,
  title={TrueMuse: A Benchmark for Data Attribution in Text-to-Music Models},
  author={Yu, Jiawei and Liu, Jian},
  journal={arXiv preprint arXiv:2610.00835},
  year={2026}
}
```
