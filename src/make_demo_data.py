from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled synthetic paired data for false-negative-aware contrastive learning."
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "clustered", "noisy"],
        default="clean",
        help="Synthetic data mode.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of paired samples.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=20,
        help="Number of semantic clusters.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Latent semantic dimension.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
        help="Feature dimension for each modality.",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.25,
        help="Fraction of corrupted positive pairs in noisy mode.",
    )
    parser.add_argument(
        "--cluster-spread",
        type=float,
        default=0.30,
        help="Within-cluster latent variation. Higher means less compact clusters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def make_projection(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
    matrix = rng.normal(0.0, 1.0, size=(in_dim, out_dim))
    matrix = matrix / np.sqrt(in_dim)
    return matrix


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def generate_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    if args.n_clusters <= 1:
        raise ValueError("--n-clusters must be greater than 1.")

    if args.n_samples < args.n_clusters:
        raise ValueError("--n-samples must be >= --n-clusters.")

    if not 0.0 <= args.noise_rate <= 1.0:
        raise ValueError("--noise-rate must be between 0 and 1.")

    rng = np.random.default_rng(args.seed)

    sample_ids = np.arange(args.n_samples)

    # Balanced cluster assignment.
    cluster_ids = np.arange(args.n_samples) % args.n_clusters
    rng.shuffle(cluster_ids)

    cluster_centers = rng.normal(0.0, 1.0, size=(args.n_clusters, args.latent_dim))
    cluster_centers = normalize_rows(cluster_centers)

    latent = (
        cluster_centers[cluster_ids]
        + rng.normal(0.0, args.cluster_spread, size=(args.n_samples, args.latent_dim))
    )
    latent = normalize_rows(latent)

    proj_a = make_projection(rng, args.latent_dim, args.feature_dim)
    proj_b = make_projection(rng, args.latent_dim, args.feature_dim)

    # Mode controls false-negative risk and modality noise.
    if args.mode == "clean":
        modality_noise = 0.10
    elif args.mode == "clustered":
        modality_noise = 0.18
    elif args.mode == "noisy":
        modality_noise = 0.18
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    a_features = latent @ proj_a + rng.normal(0.0, modality_noise, size=(args.n_samples, args.feature_dim))
    b_features_clean = latent @ proj_b + rng.normal(0.0, modality_noise, size=(args.n_samples, args.feature_dim))

    positive_partner_id = sample_ids.copy()
    is_corrupted_pair = np.zeros(args.n_samples, dtype=int)

    if args.mode == "noisy":
        n_corrupt = int(round(args.noise_rate * args.n_samples))
        corrupt_positions = rng.choice(args.n_samples, size=n_corrupt, replace=False)

        shuffled_partners = positive_partner_id[corrupt_positions].copy()
        rng.shuffle(shuffled_partners)

        positive_partner_id[corrupt_positions] = shuffled_partners
        is_corrupted_pair[corrupt_positions] = (
            positive_partner_id[corrupt_positions] != sample_ids[corrupt_positions]
        ).astype(int)

    b_features = b_features_clean[positive_partner_id]

    a_features = normalize_rows(a_features)
    b_features = normalize_rows(b_features)

    rows = []
    for i in range(args.n_samples):
        row = {
            "sample_id": int(sample_ids[i]),
            "cluster_id": int(cluster_ids[i]),
            "positive_partner_id": int(positive_partner_id[i]),
            "is_corrupted_pair": int(is_corrupted_pair[i]),
        }

        for j in range(args.feature_dim):
            row[f"a_{j:03d}"] = float(a_features[i, j])
            row[f"b_{j:03d}"] = float(b_features[i, j])

        rows.append(row)

    df = pd.DataFrame(rows)

    metadata = {
        "mode": args.mode,
        "n_samples": args.n_samples,
        "n_clusters": args.n_clusters,
        "latent_dim": args.latent_dim,
        "feature_dim": args.feature_dim,
        "noise_rate": args.noise_rate if args.mode == "noisy" else 0.0,
        "cluster_spread": args.cluster_spread,
        "seed": args.seed,
        "description": (
            "Controlled synthetic paired data for studying standard InfoNCE "
            "versus false-negative-aware InfoNCE."
        ),
    }

    return df, metadata


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data_demo"
    data_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = generate_data(args)

    data_path = data_dir / "demo_pairs.csv"
    metadata_path = data_dir / "demo_metadata.json"

    df.to_csv(data_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Demo data written to: {data_path}")
    print(f"Metadata written to : {metadata_path}")
    print(f"Mode                : {metadata['mode']}")
    print(f"Samples             : {metadata['n_samples']}")
    print(f"Clusters            : {metadata['n_clusters']}")
    print(f"Feature dim         : {metadata['feature_dim']}")
    print(f"Corrupted pairs     : {int(df['is_corrupted_pair'].sum())}")


if __name__ == "__main__":
    main()