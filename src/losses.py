from __future__ import annotations

import torch
import torch.nn.functional as F


def standard_symmetric_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Standard symmetric InfoNCE.

    Each sample treats its exact paired sample as positive.
    Every other sample in the batch is treated as a negative.
    """
    batch_size = z_a.size(0)

    logits = z_a @ z_b.T / temperature
    labels = torch.arange(batch_size, device=z_a.device)

    loss_a_to_b = F.cross_entropy(logits, labels)
    loss_b_to_a = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss_a_to_b + loss_b_to_a)

    with torch.no_grad():
        pos_sim = (z_a * z_b).sum(dim=1).mean().item()

        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_a.device)
        neg_sim = (z_a @ z_b.T)[neg_mask].mean().item()

    return loss, {
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
    }


def fn_aware_symmetric_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    cluster_ids: torch.Tensor,
    temperature: float = 0.10,
    alpha: float = 0.50,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    False-negative-aware symmetric InfoNCE.

    Samples from the same semantic cluster are not treated as hard negatives.
    Their negative contribution is downweighted by alpha.

    alpha = 0.0 means same-cluster negatives are almost ignored.
    alpha = 1.0 behaves closer to standard InfoNCE.
    """
    batch_size = z_a.size(0)
    device = z_a.device

    sim = z_a @ z_b.T
    logits = sim / temperature

    labels = torch.arange(batch_size, device=device)

    same_cluster = cluster_ids[:, None] == cluster_ids[None, :]
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)

    # Weight matrix for denominator.
    # True positive stays weight 1.
    # Different-cluster negatives stay weight 1.
    # Same-cluster non-paired negatives get alpha.
    weights = torch.ones_like(logits)
    weights = torch.where(same_cluster & ~eye, torch.full_like(weights, alpha), weights)

    loss_a_to_b = weighted_cross_entropy(logits, labels, weights)
    loss_b_to_a = weighted_cross_entropy(logits.T, labels, weights.T)
    loss = 0.5 * (loss_a_to_b + loss_b_to_a)

    with torch.no_grad():
        pos_sim = sim.diagonal().mean().item()

        neg_mask = ~eye
        same_cluster_neg_mask = same_cluster & ~eye
        diff_cluster_neg_mask = ~same_cluster

        neg_sim = sim[neg_mask].mean().item()
        same_cluster_neg_sim = (
            sim[same_cluster_neg_mask].mean().item()
            if same_cluster_neg_mask.any()
            else float("nan")
        )
        diff_cluster_neg_sim = (
            sim[diff_cluster_neg_mask].mean().item()
            if diff_cluster_neg_mask.any()
            else float("nan")
        )

    return loss, {
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "same_cluster_neg_sim": same_cluster_neg_sim,
        "diff_cluster_neg_sim": diff_cluster_neg_sim,
        "alpha": float(alpha),
    }


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Cross entropy where denominator terms can be weighted.

    For each row i:
        loss_i = -log( exp(logit_i,pos) / sum_j weight_ij * exp(logit_ij) )

    The positive term should have weight 1.
    """
    max_logits = logits.max(dim=1, keepdim=True).values
    stable_logits = logits - max_logits

    exp_logits = torch.exp(stable_logits) * weights
    denominator = exp_logits.sum(dim=1)

    positive_logits = stable_logits[torch.arange(logits.size(0), device=logits.device), labels]
    loss = -positive_logits + torch.log(denominator + 1e-12)

    return loss.mean()