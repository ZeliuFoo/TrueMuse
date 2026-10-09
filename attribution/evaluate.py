"""Evaluate finetuned attribution models.

Normal eval:
  python evaluate.py --ckpt-dir checkpoints --out-dir results

Cross-generator eval (train on gen A, test queries from gen B):
  python evaluate.py --ckpt-dir checkpoints --out-dir results --test-generator all
  python evaluate.py --ckpt-dir checkpoints --out-dir results --test-generator mustango
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset import ENCODERS, GENERATORS, EMB_ROOT, build_split, get_split
from model import AttributionModel

OUT_ROOT     = Path("results")
JAMENDO_ROOT = Path("/path/to/truemuse-data/data/embeddings/jamendo")  # update to <dataset_root>/data/embeddings/jamendo
QUERY_CHUNK  = 512   # queries per chunk; larger is faster on GPU

# ── metrics ───────────────────────────────────────────────────────────────────

PROMPT_TO_SLICE = {
    "contextual": ["contextual"],
    "style":      ["style"],
    "both":       ["all"],
}

def compute_metrics(g_emb, g_cpt, q_emb, q_cpt, q_pt, ks=(1, 5, 10, 100),
                    train_prompt="both", device=None):
    max_k     = max(ks)
    Q         = len(q_emb)
    pos_counts = np.zeros(Q, dtype=np.int32)
    topk_hits  = np.zeros((Q, max_k), dtype=bool)
    aps        = np.zeros(Q)

    # encode string labels as int32 for fast comparison
    all_labels = sorted(set(g_cpt) | set(q_cpt))
    lid        = {l: i for i, l in enumerate(all_labels)}
    g_ids      = np.array([lid[l] for l in g_cpt], dtype=np.int32)
    q_ids      = np.array([lid[l] for l in q_cpt], dtype=np.int32)

    use_gpu = device is not None and device.type != "cpu"
    if use_gpu:
        g_t = torch.from_numpy(g_emb).to(device)

    for start in range(0, Q, QUERY_CHUNK):
        end = min(start + QUERY_CHUNK, Q)
        if use_gpu:
            q_t   = torch.from_numpy(q_emb[start:end]).to(device)
            order = torch.argsort(-(q_t @ g_t.T), dim=1).cpu().numpy()
        else:
            order = np.argsort(-(q_emb[start:end] @ g_emb.T), axis=1)

        for i in range(end - start):
            ranked_ids         = g_ids[order[i]]
            match_row          = ranked_ids == q_ids[start + i]
            pos_counts[start+i] = match_row.sum()
            n_fill              = min(max_k, len(match_row))
            topk_hits[start+i, :n_fill] = match_row[:n_fill]
            hits = np.where(match_row)[0] + 1
            aps[start+i] = 0.0 if len(hits) == 0 else float(
                (np.arange(1, len(hits)+1) / hits).mean()
            )

    allowed_slices = PROMPT_TO_SLICE.get(train_prompt, ["all", "contextual", "style"])
    all_slices = [("all",        np.ones(Q, bool)),
                  ("contextual", q_pt == "contextual"),
                  ("style",      q_pt == "style")]

    rows = []
    for slice_name, mask in all_slices:
        if slice_name not in allowed_slices:
            continue
        n = mask.sum()
        if n == 0:
            continue
        pos_m  = pos_counts[mask]
        hits_m = topk_hits[mask]
        row    = {"slice": slice_name, "n": int(n)}
        for k in ks:
            row[f"R@{k}"] = float(
                np.where(pos_m > 0, hits_m[:, :k].sum(1) / pos_m, 0.0).mean()
            )
        row["mAP"] = float(aps[mask].mean())
        rows.append(row)
    return rows


# ── model helpers ─────────────────────────────────────────────────────────────

APPLY_CHUNK = 2048   # rows per batch when mapping large embedding arrays

@torch.no_grad()
def apply_model(model, emb, device, mode):
    fn = model.map_exemplar if mode == "exemplar" else model.map_query
    if len(emb) <= APPLY_CHUNK:
        t = torch.tensor(emb, dtype=torch.float32, device=device)
        return fn(t).cpu().numpy()
    out = []
    for start in range(0, len(emb), APPLY_CHUNK):
        t = torch.tensor(emb[start:start+APPLY_CHUNK], dtype=torch.float32, device=device)
        out.append(fn(t).cpu().numpy())
    return np.concatenate(out)


_jamendo_raw_cache: dict[str, np.ndarray] = {}

def load_jamendo_raw(encoder: str) -> np.ndarray | None:
    if encoder not in _jamendo_raw_cache:
        # Only keep one encoder in RAM at a time to avoid OOM
        _jamendo_raw_cache.clear()
        pool_path = JAMENDO_ROOT / encoder / "pool.npz"
        if not pool_path.exists():
            _jamendo_raw_cache[encoder] = None
        else:
            _jamendo_raw_cache[encoder] = np.load(pool_path)["emb"].astype(np.float32)
    return _jamendo_raw_cache[encoder]

def load_jamendo_distractors(encoder, model, device):
    emb = load_jamendo_raw(encoder)
    if emb is None:
        return None
    return apply_model(model, emb, device, "exemplar")


# ── cross-generator data loading ──────────────────────────────────────────────

def build_crossgen_test_data(encoder, train_gen, test_gen, tasks):
    """Exemplars from train_gen, queries from test_gen. Concept split from train_gen."""
    g_embs, g_cpts = [], []
    q_embs, q_cpts, q_pts = [], [], []

    for task in tasks:
        test_concepts = get_split(encoder, train_gen, task)[2]

        for c in sorted(test_concepts):
            label    = f"{task}/{c}"
            ex_path  = EMB_ROOT / encoder / train_gen / task / c / "exemplars.npz"
            q_path   = EMB_ROOT / encoder / test_gen  / task / c / "queries.npz"
            if not ex_path.exists() or not q_path.exists():
                continue

            ex = np.load(ex_path)
            qu = np.load(q_path)

            g_embs.append(ex["emb"].astype(np.float32))
            g_cpts.extend([label] * len(ex["emb"]))
            q_embs.append(qu["emb"].astype(np.float32))
            q_cpts.extend([label] * len(qu["emb"]))
            q_pts.extend(qu["prompt_type"].tolist())

    if not g_embs:
        return None
    return (
        np.concatenate(g_embs), np.array(g_cpts),
        np.concatenate(q_embs), np.array(q_cpts),
        np.array(q_pts),
    )


# ── per-checkpoint evaluation ─────────────────────────────────────────────────

def _eval_one(model, device, encoder, train_gen, tasks_str, prompt,
              mapper_type, lam_reg, test_data, extra_fields):
    """Shared logic: map embeddings, add distractors, compute per-task metrics."""
    if test_data is None:
        return []

    g_emb, g_cpt, q_emb, q_cpt, q_pt = test_data
    g_mapped = apply_model(model, g_emb, device, "exemplar")
    q_mapped = apply_model(model, q_emb, device, "query")

    jamendo = load_jamendo_distractors(encoder, model, device)
    n_dist  = 0
    if jamendo is not None:
        n_dist   = len(jamendo)
        g_mapped = np.concatenate([g_mapped, jamendo])
        g_cpt    = np.concatenate([g_cpt, np.array(["__distractor__"] * n_dist)])

    all_rows = []
    eval_tasks = sorted(set(c.split("/")[0] for c in g_cpt if c != "__distractor__"))
    for task in eval_tasks:
        g_mask = np.array([c.startswith(f"{task}/") or c == "__distractor__" for c in g_cpt])
        q_mask = np.array([c.startswith(f"{task}/") for c in q_cpt])
        if g_mask.sum() == 0 or q_mask.sum() == 0:
            continue
        rows = compute_metrics(
            g_mapped[g_mask], g_cpt[g_mask],
            q_mapped[q_mask], q_cpt[q_mask], q_pt[q_mask],
            train_prompt=prompt, device=device,
        )
        for r in rows:
            r.update({
                "encoder": encoder, "train_generator": train_gen,
                "train_tasks": tasks_str, "eval_task": task,
                "train_prompt": prompt,
                "n_distractors": n_dist,
                "mapper_type": mapper_type,
                "lam_reg": lam_reg,
            })
            r.update(extra_fields)
        all_rows.extend(rows)
    return all_rows


def evaluate_checkpoint(ckpt_path, device, test_generator=None):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AttributionModel(ckpt["d_in"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    encoder    = ckpt["encoder"]
    train_gen  = ckpt["generator"]
    tasks      = ckpt["tasks"].split(",")
    tasks_str  = ckpt["tasks"]
    mapper_type = ckpt.get("mapper_type", "residual")
    lam_reg     = ckpt.get("lam_reg", 0.05)

    if test_generator is None:
        test_data = build_split(encoder, train_gen, tasks, "test")
        return _eval_one(model, device, encoder, train_gen, tasks_str,
                         ckpt["prompt"], mapper_type, lam_reg,
                         test_data, {"test_generator": train_gen})
    else:
        test_gens = GENERATORS if test_generator == "all" else [test_generator]
        all_rows  = []
        for tgen in test_gens:
            test_data = build_crossgen_test_data(encoder, train_gen, tgen, tasks)
            rows = _eval_one(model, device, encoder, train_gen, tasks_str,
                             ckpt["prompt"], mapper_type, lam_reg,
                             test_data, {"test_generator": tgen})
            all_rows.extend(rows)
        return all_rows


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(all_rows):
    numeric_keys = ["R@1", "R@5", "R@10", "R@100", "mAP"]
    group = defaultdict(list)
    for r in all_rows:
        key = (r["encoder"], r["train_generator"], r.get("test_generator", ""),
               r["train_tasks"], r["train_prompt"],
               r["eval_task"], r["slice"],
               r["mapper_type"], str(r["lam_reg"]))
        group[key].append(r)

    aggregated = []
    for _, rows in group.items():
        base = {k: v for k, v in rows[0].items()
                if k not in numeric_keys + ["n"]}
        base["n_folds"] = len(rows)
        base["n"]       = int(np.mean([r["n"] for r in rows]))
        for m in numeric_keys:
            vals = [r[m] for r in rows]
            base[m]          = float(np.mean(vals))
            base[f"{m}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        aggregated.append(base)
    return aggregated


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir",       default="checkpoints")
    ap.add_argument("--out-dir",        default=str(OUT_ROOT))
    ap.add_argument("--pattern",        default="*.pt")
    ap.add_argument("--test-generator", default=None,
                    help="cross-generator eval: generator name or 'all'. "
                         "Omit for normal same-generator eval.")
    args = ap.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_keys = ["R@1", "R@5", "R@10", "R@100", "mAP"]
    fieldnames = (
        ["encoder", "train_generator", "test_generator",
         "train_tasks", "train_prompt", "eval_task",
         "slice", "n", "n_distractors", "mapper_type", "lam_reg"]
        + numeric_keys
    )

    fname    = "crossgen_results.csv" if args.test_generator else "finetune_results.csv"
    out_path = out_dir / fname
    total    = 0

    # Build set of already-evaluated checkpoints from existing file
    done = set()
    if out_path.exists():
        with open(out_path, newline="") as ef:
            for row in csv.DictReader(ef):
                done.add((row["encoder"], row["train_generator"],
                          row["train_tasks"], row["train_prompt"]))
        print(f"Resuming: {len(done)} checkpoint configs already done, skipping.", flush=True)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()

        for ckpt_path in sorted(ckpt_dir.glob(args.pattern)):
            # parse key from filename: encoder__generator__tasks__prompt.pt
            stem = ckpt_path.stem
            parts = stem.split("__")
            # cross-gen: only single-task + both
            if args.test_generator:
                is_single = "-" not in parts[2]   # no comma→hyphen means single task
                if not is_single or parts[3] != "both":
                    continue
            key = (parts[0], parts[1], parts[2].replace("-", ","), parts[3])
            if key in done:
                continue
            print(f"Evaluating {ckpt_path.name} …", flush=True)
            rows = evaluate_checkpoint(ckpt_path, device, args.test_generator)
            w.writerows(rows)
            f.flush()
            total += len(rows)
            print(f"  → {len(rows)} rows", flush=True)

    print(f"\nWrote {total} rows → {out_path}")


if __name__ == "__main__":
    main()
