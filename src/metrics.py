from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def retrieval_metrics(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sample_ids: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """
    Evaluate A-to-B retrieval.

    The correct B item for each A item has the same sample_id.
    """
    z_a = z_a.float().cpu()
    z_b = z_b.float().cpu()

    sim = (z_a @ z_b.T).numpy()
    n_a, n_b = sim.shape

    if n_a != len(sample_ids) or n_b != len(sample_ids):
        raise ValueError("Embedding count must match sample_ids length.")

    max_k = min(max(k_values), n_b)

    top_part = np.argpartition(sim, -max_k, axis=1)[:, -max_k:]
    top_scores = sim[np.arange(n_a)[:, None], top_part]
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top_sorted = top_part[np.arange(n_a)[:, None], order]

    metrics: dict[str, float] = {}

    for k_req in k_values:
        k = min(k_req, n_b)

        retrieved_ids = sample_ids[top_sorted[:, :k]]
        hit = (retrieved_ids == sample_ids[:, None]).any(axis=1)

        recall = float(hit.mean())
        random_recall = float(min(k / n_b, 1.0))

        metrics[f"recall@{k_req}"] = recall
        metrics[f"lift@{k_req}"] = (
            float(recall / random_recall) if random_recall > 0 else math.inf
        )

    pos_sim = sim.diagonal()
    metrics["pos_sim_mean"] = float(pos_sim.mean())
    metrics["n_pool"] = float(n_b)

    return metrics


@torch.no_grad()
def cluster_similarity_metrics(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    cluster_ids: np.ndarray,
) -> dict[str, float]:
    """
    Measure average similarity for:
    - true pairs
    - same-cluster non-pairs
    - different-cluster pairs
    """
    z_a = z_a.float().cpu()
    z_b = z_b.float().cpu()

    sim = (z_a @ z_b.T).numpy()
    n = sim.shape[0]

    same_cluster = cluster_ids[:, None] == cluster_ids[None, :]
    eye = np.eye(n, dtype=bool)

    same_cluster_neg = same_cluster & ~eye
    diff_cluster_neg = ~same_cluster

    result = {
        "eval_pos_sim": float(np.diag(sim).mean()),
        "eval_same_cluster_neg_sim": (
            float(sim[same_cluster_neg].mean()) if same_cluster_neg.any() else float("nan")
        ),
        "eval_diff_cluster_neg_sim": (
            float(sim[diff_cluster_neg].mean()) if diff_cluster_neg.any() else float("nan")
        ),
    }

    return result