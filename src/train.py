from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from losses import fn_aware_symmetric_infonce, standard_symmetric_infonce
from metrics import cluster_similarity_metrics, retrieval_metrics
from model import DualEncoder


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.10
    alpha: float = 0.50
    hidden_dim: int = 256
    embed_dim: int = 128
    dropout: float = 0.10
    val_fraction: float = 0.20
    k_values: tuple[int, ...] = (1, 5, 10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standard InfoNCE or false-negative-aware InfoNCE on synthetic paired data."
    )

    parser.add_argument("--data-csv", type=Path, default=Path("data_demo/demo_pairs.csv"))
    parser.add_argument("--metadata-json", type=Path, default=Path("data_demo/demo_metadata.json"))

    parser.add_argument(
        "--loss",
        choices=["standard", "fn_aware"],
        default="standard",
        help="Contrastive loss variant.",
    )

    parser.add_argument("--alpha", type=float, default=TrainConfig.alpha)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)

    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairedFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, a_cols: list[str], b_cols: list[str]):
        self.sample_ids = df["sample_id"].astype(int).to_numpy()
        self.cluster_ids = df["cluster_id"].astype(int).to_numpy()

        self.x_a = df[a_cols].astype(np.float32).to_numpy()
        self.x_b = df[b_cols].astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x_a[idx].copy()),
            torch.from_numpy(self.x_b[idx].copy()),
            int(self.sample_ids[idx]),
            int(self.cluster_ids[idx]),
        )


def split_dataframe(df: pd.DataFrame, seed: int, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_val = max(1, int(round(len(indices) * val_fraction)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df


def collate_batch(batch):
    x_a = torch.stack([item[0] for item in batch])
    x_b = torch.stack([item[1] for item in batch])
    sample_ids = torch.tensor([item[2] for item in batch], dtype=torch.long)
    cluster_ids = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return x_a, x_b, sample_ids, cluster_ids


@torch.no_grad()
def embed_dataset(model: DualEncoder, loader: DataLoader, device: torch.device):
    model.eval()

    z_a_all = []
    z_b_all = []
    sample_ids_all = []
    cluster_ids_all = []

    for x_a, x_b, sample_ids, cluster_ids in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)

        z_a = model.encode_a(x_a)
        z_b = model.encode_b(x_b)

        z_a_all.append(z_a.cpu())
        z_b_all.append(z_b.cpu())
        sample_ids_all.extend(sample_ids.numpy().tolist())
        cluster_ids_all.extend(cluster_ids.numpy().tolist())

    return (
        torch.cat(z_a_all, dim=0),
        torch.cat(z_b_all, dim=0),
        np.asarray(sample_ids_all, dtype=np.int64),
        np.asarray(cluster_ids_all, dtype=np.int64),
    )


def evaluate(model: DualEncoder, loader: DataLoader, device: torch.device, k_values: tuple[int, ...]) -> dict[str, float]:
    z_a, z_b, sample_ids, cluster_ids = embed_dataset(model, loader, device)

    out = {}
    out.update(retrieval_metrics(z_a, z_b, sample_ids, k_values=k_values))
    out.update(cluster_similarity_metrics(z_a, z_b, cluster_ids))

    return out


def train_one_epoch(
    model: DualEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    temperature: float,
    alpha: float,
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    running_pos_sim = 0.0
    running_neg_sim = 0.0
    running_same_cluster_neg_sim = 0.0
    running_diff_cluster_neg_sim = 0.0
    counted_same_diff = 0
    steps = 0

    for x_a, x_b, _sample_ids, cluster_ids in tqdm(loader, desc="train", leave=False):
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        cluster_ids = cluster_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        z_a, z_b = model(x_a, x_b)

        if loss_name == "standard":
            loss, loss_metrics = standard_symmetric_infonce(
                z_a=z_a,
                z_b=z_b,
                temperature=temperature,
            )
        elif loss_name == "fn_aware":
            loss, loss_metrics = fn_aware_symmetric_infonce(
                z_a=z_a,
                z_b=z_b,
                cluster_ids=cluster_ids,
                temperature=temperature,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu())
        running_pos_sim += float(loss_metrics.get("pos_sim", 0.0))
        running_neg_sim += float(loss_metrics.get("neg_sim", 0.0))

        if "same_cluster_neg_sim" in loss_metrics and not math.isnan(loss_metrics["same_cluster_neg_sim"]):
            running_same_cluster_neg_sim += float(loss_metrics["same_cluster_neg_sim"])
            running_diff_cluster_neg_sim += float(loss_metrics["diff_cluster_neg_sim"])
            counted_same_diff += 1

        steps += 1

    result = {
        "train_loss": running_loss / max(steps, 1),
        "train_pos_sim": running_pos_sim / max(steps, 1),
        "train_neg_sim": running_neg_sim / max(steps, 1),
    }

    if counted_same_diff > 0:
        result["train_same_cluster_neg_sim"] = running_same_cluster_neg_sim / counted_same_diff
        result["train_diff_cluster_neg_sim"] = running_diff_cluster_neg_sim / counted_same_diff
    else:
        result["train_same_cluster_neg_sim"] = float("nan")
        result["train_diff_cluster_neg_sim"] = float("nan")

    return result


def save_checkpoint(path: Path, model: DualEncoder, optimizer: torch.optim.Optimizer, epoch: int, row: dict, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_path = resolve_path(args.data_csv, repo_root)
    metadata_path = resolve_path(args.metadata_json, repo_root)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data CSV: {data_path}")

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    mode = metadata.get("mode", "unknown")
    n_samples = metadata.get("n_samples", "unknown")

    run_name = f"{mode}_{n_samples}_{args.loss}"
    if args.loss == "fn_aware":
        run_name += f"_alpha{args.alpha}"

    output_dir = args.output_dir or (repo_root / "outputs" / run_name)
    checkpoint_dir = args.checkpoint_dir or (repo_root / "checkpoints" / run_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    set_seed(cfg.seed)

    df = pd.read_csv(data_path)

    required = ["sample_id", "cluster_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing}. Available: {df.columns.tolist()}")

    a_cols = sorted([c for c in df.columns if c.startswith("a_")])
    b_cols = sorted([c for c in df.columns if c.startswith("b_")])

    if len(a_cols) == 0 or len(a_cols) != len(b_cols):
        raise RuntimeError("Could not find matching a_* and b_* feature columns.")

    train_df, val_df = split_dataframe(df, seed=cfg.seed, val_fraction=cfg.val_fraction)

    train_ds = PairedFeatureDataset(train_df, a_cols, b_cols)
    val_ds = PairedFeatureDataset(val_df, a_cols, b_cols)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualEncoder(
        input_dim=len(a_cols),
        hidden_dim=cfg.hidden_dim,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    config_dump = {
        **asdict(cfg),
        "loss": args.loss,
        "mode": mode,
        "metadata": metadata,
        "data_csv": str(data_path),
        "feature_dim": len(a_cols),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "device": str(device),
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    metrics_path = output_dir / "metrics.csv"
    history: list[dict] = []

    print(f"Mode          : {mode}")
    print(f"Samples       : {len(df)}")
    print(f"Train / Val   : {len(train_df)} / {len(val_df)}")
    print(f"Loss          : {args.loss}")
    print(f"Alpha         : {args.alpha}")
    print(f"Feature dim   : {len(a_cols)}")
    print(f"Device        : {device}")
    print(f"Output dir    : {output_dir}")

    best_lift50 = -math.inf

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_name=args.loss,
            temperature=cfg.temperature,
            alpha=cfg.alpha,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            k_values=cfg.k_values,
        )

        row = {
            "epoch": float(epoch),
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)

        print(
            f"loss={row['train_loss']:.4f} "
            f"R@10={row['val_recall@10']:.4f} "
            f"R@50={row['val_recall@50']:.4f} "
            f"Lift@50={row['val_lift@50']:.2f}x "
            f"pos_sim={row['val_pos_sim_mean']:.4f}"
        )

        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row, config_dump)

        if row["val_lift@50"] > best_lift50:
            best_lift50 = row["val_lift@50"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, config_dump)

    print("\nDone.")
    print(f"Best Lift@50 : {best_lift50:.2f}x")
    print(f"Metrics      : {metrics_path}")


if __name__ == "__main__":
    main()