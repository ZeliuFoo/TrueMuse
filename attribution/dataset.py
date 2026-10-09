"""Data loading and train/val/test splitting for attribution finetuning."""

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

EMB_ROOT = Path("/path/to/truemuse-data/embeddings")  # update to <dataset_root>/embeddings

ENCODERS   = ["mert-v0", "mert-v0-public", "music2vec-v1", "dac-16k"]
GENERATORS = ["audioldm2", "mustango", "stable-audio"]
TASKS      = ["musician_3", "musician_6", "instrument_3", "instrument_6", "genre_3", "genre_6", "melody_1"]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SPLIT_SEED  = 42


def get_concepts(encoder, generator, task):
    root = EMB_ROOT / encoder / generator / task
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def get_shared_concepts(generator, task):
    """Intersection of concepts across ALL encoders AND ALL generators for a task.
    Ensures train/val/test split is identical across every encoder/generator combo."""
    sets = []
    for enc in ENCODERS:
        for gen in GENERATORS:
            root = EMB_ROOT / enc / gen / task
            if root.exists():
                sets.append(set(d.name for d in root.iterdir() if d.is_dir()))
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def get_split(encoder, generator, task, seed=SPLIT_SEED):
    """Returns (train_concepts, val_concepts, test_concepts) using 70/15/15 split."""
    concepts = get_shared_concepts(generator, task)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(concepts))
    n_train = int(len(concepts) * TRAIN_RATIO)
    n_val   = int(len(concepts) * VAL_RATIO)

    train = [concepts[i] for i in idx[:n_train]]
    val   = [concepts[i] for i in idx[n_train:n_train + n_val]]
    test  = [concepts[i] for i in idx[n_train + n_val:]]
    return train, val, test


def load_embeddings(encoder, generator, task, concepts):
    """Load exemplar + query embeddings for the given concept subset."""
    root = EMB_ROOT / encoder / generator / task
    concept_set = set(concepts)

    g_emb, g_cpt = [], []
    q_emb, q_cpt, q_pt = [], [], []

    for c in sorted(concept_set):
        cdir = root / c
        if not cdir.is_dir():
            continue
        ex = np.load(cdir / "exemplars.npz")
        qu = np.load(cdir / "queries.npz")

        label = f"{task}/{c}"
        g_emb.append(ex["emb"].astype(np.float32))
        g_cpt.extend([label] * len(ex["emb"]))

        q_emb.append(qu["emb"].astype(np.float32))
        q_cpt.extend([label] * len(qu["emb"]))
        q_pt.extend(qu["prompt_type"].tolist())

    if not g_emb:
        return None

    return (
        np.concatenate(g_emb),
        np.array(g_cpt),
        np.concatenate(q_emb),
        np.array(q_cpt),
        np.array(q_pt),
    )


def build_split(encoder, generator, tasks, split):
    """
    Merge embeddings across tasks for a given split ('train'/'val'/'test').
    Returns (g_emb, g_cpt, q_emb, q_cpt, q_pt) or None if empty.
    """
    parts = []
    for task in tasks:
        tr, va, te = get_split(encoder, generator, task)
        concepts = {"train": tr, "val": va, "test": te}[split]
        if not concepts:
            continue

        data = load_embeddings(encoder, generator, task, concepts)
        if data is not None:
            parts.append(data)

    if not parts:
        return None

    return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))


class AttributionDataset(Dataset):
    """
    One (exemplar, query) pair per concept per epoch.
    Queries are optionally filtered by prompt_type.
    """

    def __init__(self, g_emb, g_cpt, q_emb, q_cpt, q_pt,
                 prompt_filter=None, seed=0):
        self.seed  = seed
        self.epoch = 0

        if prompt_filter is not None:
            mask  = np.isin(q_pt, prompt_filter)
            q_emb = q_emb[mask]
            q_cpt = q_cpt[mask]

        concepts = sorted(set(g_cpt))
        self.g_by = {c: g_emb[g_cpt == c] for c in concepts}
        self.q_by = {c: q_emb[q_cpt == c] for c in concepts}
        self.concepts = [c for c in concepts
                         if len(self.g_by[c]) > 0 and len(self.q_by[c]) > 0]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        c   = self.concepts[idx]
        rng = np.random.default_rng(self.seed + self.epoch * 100_000 + idx)
        g   = self.g_by[c][rng.integers(len(self.g_by[c]))]
        q   = self.q_by[c][rng.integers(len(self.q_by[c]))]
        return g, q
