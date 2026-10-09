"""Train bottleneck attribution model with NT-Xent loss + weight decay."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    ENCODERS, GENERATORS, EMB_ROOT,
    get_concepts, get_split, build_split, AttributionDataset,
)
from model import AttributionModel

TEMPERATURE  = 1.0
JAMENDO_ROOT = Path("/path/to/truemuse-data/data/embeddings/jamendo")  # update to <dataset_root>/data/embeddings/jamendo


# ── loss ──────────────────────────────────────────────────────────────────────

def nt_xent(a: torch.Tensor, b: torch.Tensor, temp: float = TEMPERATURE):
    """Symmetric NT-Xent.  a, b: (B, D) L2-normalized."""
    sim    = a @ b.T / temp          # (B, B)
    labels = torch.arange(len(a), device=a.device)
    return (nn.functional.cross_entropy(sim, labels) +
            nn.functional.cross_entropy(sim.T, labels)) / 2


# ── quick eval on numpy arrays ────────────────────────────────────────────────

def recall_at_k(g_emb, g_cpt, q_emb, q_cpt, ks=(1, 5, 10, 100)):
    sims   = q_emb @ g_emb.T                        # (Q, G)
    order  = np.argsort(-sims, axis=1)
    ranked = g_cpt[order]                            # (Q, G)
    match  = ranked == q_cpt[:, None]               # (Q, G)
    pos    = match.sum(axis=1)                       # positives per query

    out = {}
    for k in ks:
        out[f"R@{k}"] = float(
            np.where(pos > 0, match[:, :k].sum(axis=1) / pos, 0.0).mean()
        )
    # mAP
    aps = []
    for i in range(len(q_cpt)):
        hits = np.where(match[i])[0] + 1
        if len(hits) == 0:
            aps.append(0.0)
        else:
            aps.append(float((np.arange(1, len(hits) + 1) / hits).mean()))
    out["mAP"] = float(np.mean(aps))
    return out


JAMENDO_POOL_SIZE = 50

def load_jamendo_raw(encoder: str) -> np.ndarray | None:
    p = JAMENDO_ROOT / encoder / "pool.npz"
    if not p.exists():
        return None
    emb = np.load(p)["emb"].astype(np.float32)
    if len(emb) > JAMENDO_POOL_SIZE:
        idx = np.random.default_rng(0).choice(len(emb), JAMENDO_POOL_SIZE, replace=False)
        emb = emb[idx]
    return emb


def eval_pretrained(data, jamendo_raw=None, ks=(1, 5, 10)):
    """Raw cosine similarity (no model), i.e. pre-trained baseline."""
    if data is None:
        return {}
    g_emb, g_cpt, q_emb, q_cpt, _ = data
    g, q = g_emb.copy(), q_emb.copy()
    if jamendo_raw is not None:
        g     = np.concatenate([g, jamendo_raw], axis=0)
        g_cpt = np.concatenate([g_cpt, np.array(["__distractor__"] * len(jamendo_raw))])
    return recall_at_k(g, g_cpt, q, q_cpt, ks)


@torch.no_grad()
def evaluate(model, data, device, jamendo_raw=None, ks=(1, 5, 10)):
    if data is None:
        return {}
    g_emb, g_cpt, q_emb, q_cpt, _ = data
    model.eval()
    g = model.map_exemplar(torch.tensor(g_emb, device=device)).cpu().numpy()
    q = model.map_query(  torch.tensor(q_emb, device=device)).cpu().numpy()

    if jamendo_raw is not None:
        j = model.map_exemplar(torch.tensor(jamendo_raw, device=device)).cpu().numpy()
        g     = np.concatenate([g, j], axis=0)
        g_cpt = np.concatenate([g_cpt, np.array(["__distractor__"] * len(j))])

    return recall_at_k(g, g_cpt, q, q_cpt, ks)


# ── training for a single (task-combo) ───────────────────────────────────────

def train_one(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks  = args.tasks.split(",")
    pfilter = None if args.prompt == "both" else [args.prompt]

    # infer embedding dim from first task / first concept
    first_task = tasks[0]
    c0   = get_concepts(args.encoder, args.generator, first_task)[0]
    d_in = np.load(
        EMB_ROOT / args.encoder / args.generator / first_task / c0 / "exemplars.npz"
    )["emb"].shape[-1]

    model     = AttributionModel(d_in).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    train_data   = build_split(args.encoder, args.generator, tasks, "train")
    val_data     = build_split(args.encoder, args.generator, tasks, "val")
    jamendo_raw  = load_jamendo_raw(args.encoder)

    if train_data is None:
        raise RuntimeError("No training data found.")

    g_emb_tr, g_cpt_tr, q_emb_tr, q_cpt_tr, q_pt_tr = train_data

    def _summary(data, name):
        if data is None:
            return f"{name}: None"
        g, gc, q, qc, qt = data
        n_concepts = len(set(gc))
        ctx = (qt == "contextual").sum()
        sty = (qt == "style").sum()
        return (f"{name}: {n_concepts} concepts | "
                f"{len(g)} exemplars | {len(q)} queries (ctx={ctx} sty={sty})")

    print(f"  {_summary(train_data, 'train')}", flush=True)
    print(f"  {_summary(val_data,   'val  ')}", flush=True)
    if jamendo_raw is not None:
        print(f"  jamendo distractors: {len(jamendo_raw)}", flush=True)

    pt_val = eval_pretrained(val_data, jamendo_raw=jamendo_raw)
    print(f"  pretrained val R@10 = {pt_val.get('R@10', float('nan')):.4f}", flush=True)

    train_ds = AttributionDataset(
        g_emb_tr, g_cpt_tr, q_emb_tr, q_cpt_tr, q_pt_tr,
        prompt_filter=pfilter, seed=args.seed,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0,
    )

    best_score  = -float("inf")
    patience_ct = 0
    best_state  = None

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss, n = 0.0, 0
        for step, (g_b, q_b) in enumerate(loader):
            g_b, q_b = g_b.to(device), q_b.to(device)
            g_m, q_m = model(g_b, q_b)
            loss = nt_xent(g_m, q_m)
            if step % 10 == 0 and args.lam_reg > 0:
                loss = loss + args.lam_reg * model.orthogonality_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1
        scheduler.step()

        n_val_concepts = len(set(val_data[1])) if val_data is not None else 0
        val_metrics = evaluate(model, val_data, device, jamendo_raw=jamendo_raw)
        score = val_metrics.get("R@10", -total_loss / n)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            pt_r10 = pt_val.get("R@10", float("nan"))
            print(f"  epoch {epoch:3d} | loss {total_loss/n:.4f} | val R@10 {score:.4f} (PT={pt_r10:.4f}) (n_val={n_val_concepts})")

        # when val is too small for meaningful R@10, always keep latest and never stop early
        if n_val_concepts < 1:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            if score > best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ct = 0
            else:
                patience_ct += 1
                if patience_ct >= args.patience:
                    print(f"  early stop at epoch {epoch}")
                    break

    model.load_state_dict(best_state)
    return model, d_in


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder",      required=True, choices=ENCODERS)
    ap.add_argument("--generator",    required=True, choices=GENERATORS)
    ap.add_argument("--tasks",        required=True,
                    help="comma-separated tasks, e.g. musician or musician,instrument")
    ap.add_argument("--prompt",       required=True,
                    choices=["contextual", "style", "both"])
    ap.add_argument("--out-dir",      default="checkpoints")
    ap.add_argument("--lam-reg",      type=float, default=0.05)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size",   type=int,   default=32)
    ap.add_argument("--epochs",       type=int,   default=200)
    ap.add_argument("--patience",     type=int,   default=20)
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    tasks    = args.tasks.split(",")
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.encoder}__{args.generator}__{args.tasks.replace(',','-')}__{args.prompt}"
    print(f"\n=== {tag} ===")

    model, d_in = train_one(args)

    ckpt = out_dir / f"{tag}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "d_in": d_in,
        "encoder": args.encoder, "generator": args.generator,
        "tasks": args.tasks, "prompt": args.prompt,
        "lam_reg": args.lam_reg,
    }, ckpt)
    print(f"  saved → {ckpt}")


if __name__ == "__main__":
    main()
