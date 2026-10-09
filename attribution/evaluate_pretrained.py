"""Evaluate pre-trained (no finetuning) encoders on the same test pool as finetuned models.

Pool = test concepts' exemplars + Jamendo distractors (same as evaluate.py).
Output: results/pretrained_results.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ENCODERS, GENERATORS, TASKS, build_split

OUT_ROOT     = Path("results")
JAMENDO_ROOT = Path("/path/to/truemuse-data/data/embeddings/jamendo")  # update to <dataset_root>/data/embeddings/jamendo
QUERY_CHUNK  = 512   # larger chunk is fine on GPU


def load_jamendo(encoder: str) -> np.ndarray | None:
    p = JAMENDO_ROOT / encoder / "pool.npz"
    if not p.exists():
        return None
    return np.load(p)["emb"].astype(np.float32)


def compute_metrics(g_emb, g_cpt, q_emb, q_cpt, q_pt,
                    ks=(1, 5, 10, 100), device=None):
    max_k      = max(ks)
    Q          = len(q_emb)
    pos_counts = np.zeros(Q, dtype=np.int32)
    topk_hits  = np.zeros((Q, max_k), dtype=bool)
    aps        = np.zeros(Q)

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
            ranked_ids          = g_ids[order[i]]
            match_row           = ranked_ids == q_ids[start + i]
            pos_counts[start+i] = match_row.sum()
            n_fill              = min(max_k, len(match_row))
            topk_hits[start+i, :n_fill] = match_row[:n_fill]
            hits = np.where(match_row)[0] + 1
            aps[start+i] = 0.0 if len(hits) == 0 else float(
                (np.arange(1, len(hits)+1) / hits).mean()
            )

    rows = []
    for slice_name, mask in [("all",        np.ones(Q, bool)),
                              ("contextual", q_pt == "contextual"),
                              ("style",      q_pt == "style")]:
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


def evaluate_pretrained(encoder, generator, tasks, jamendo=None, device=None):
    test_data = build_split(encoder, generator, tasks, "test")
    if test_data is None:
        return []

    g_emb, g_cpt, q_emb, q_cpt, q_pt = test_data

    n_dist = 0
    if jamendo is not None:
        n_dist = len(jamendo)
        g_emb  = np.concatenate([g_emb, jamendo], axis=0)
        g_cpt  = np.concatenate([g_cpt, np.array(["__distractor__"] * n_dist)])

    all_rows = []
    eval_tasks = sorted(set(c.split("/")[0] for c in g_cpt if c != "__distractor__"))

    for task in eval_tasks:
        g_mask = np.array([c.startswith(f"{task}/") or c == "__distractor__"
                           for c in g_cpt])
        q_mask = np.array([c.startswith(f"{task}/") for c in q_cpt])
        if g_mask.sum() == 0 or q_mask.sum() == 0:
            continue

        rows = compute_metrics(
            g_emb[g_mask], g_cpt[g_mask],
            q_emb[q_mask], q_cpt[q_mask], q_pt[q_mask],
            device=device,
        )
        for r in rows:
            r.update({"encoder": encoder, "generator": generator,
                      "eval_task": task, "n_distractors": n_dist})
        all_rows.extend(rows)

    return all_rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    numeric_keys = ["R@1", "R@5", "R@10", "R@100", "mAP"]
    fieldnames   = (["encoder", "generator", "eval_task", "slice", "n", "n_distractors"]
                    + numeric_keys)

    out_path = OUT_ROOT / "pretrained_results.csv"
    total    = 0

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()

        for encoder in ENCODERS:
            print(f"Loading Jamendo for {encoder} …", flush=True)
            jamendo = load_jamendo(encoder)

            for generator in GENERATORS:
                for task in TASKS:
                    print(f"  {encoder} / {generator} / {task}", flush=True)
                    rows = evaluate_pretrained(encoder, generator, [task],
                                               jamendo=jamendo, device=device)
                    w.writerows(rows)
                    f.flush()
                    total += len(rows)

    print(f"\nWrote {total} rows → {out_path}")


if __name__ == "__main__":
    main()
