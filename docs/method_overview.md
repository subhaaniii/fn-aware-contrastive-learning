# Method Overview

## Goal

This repository studies the false-negative problem in contrastive learning.

Standard InfoNCE treats every non-matching sample in a batch as a negative. This assumption can be problematic when two samples are not exact pairs but still belong to the same semantic group.

The goal is to compare standard InfoNCE with a false-negative-aware InfoNCE variant under controlled retrieval conditions.

## Methods Compared

### Standard InfoNCE

Standard InfoNCE pulls matched pairs together and pushes all other samples in the batch apart.

This works well when negatives are truly dissimilar. However, when many samples share semantic structure, some negatives may actually be false negatives.

### False-Negative-Aware InfoNCE

The false-negative-aware variant uses synthetic semantic cluster labels.

If two non-matching samples share the same cluster, their negative penalty is reduced. This tests whether relaxing same-cluster negatives can improve retrieval when semantic overlap is present.

## Paper Inspiration

This repository is motivated by the false-negative problem in contrastive learning.

Standard InfoNCE assumes that non-matching samples in a batch are negatives. This works well when negatives are truly dissimilar, but it becomes problematic when semantically similar samples appear as negatives.

Debiased Contrastive Learning studies this negative-sampling bias and proposes a contrastive objective that corrects for the chance of sampling same-label examples as negatives. False Negative Cancellation further studies how false negatives can slow representation learning and damage semantic structure.

The controlled benchmark in this repository follows the same motivation, but keeps the setup simple and inspectable. Synthetic cluster labels are used to create known false-negative structure, and same-cluster negatives are downweighted during training.

## Experimental Setup

The repository uses synthetic paired data with two modalities:

- Modality A: synthetic feature vector
- Modality B: paired synthetic feature vector
- Cluster label: semantic group identifier

Three modes are tested:

| Mode | Purpose |
|---|---|
| Clean | Test behavior when pairs are correct and false-negative pressure is low |
| Clustered | Test behavior when many samples are semantically similar |
| Noisy | Test behavior when some positive pair assignments are corrupted |

## Main Lesson

False-negative-aware learning can help when semantic overlap creates harmful negatives, but it is not universally better than standard InfoNCE.

The effectiveness depends on cluster structure, pair quality, alpha value, and sample size.