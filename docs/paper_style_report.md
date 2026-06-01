# Paper-Style Report: False-Negative-Aware Contrastive Learning

## Abstract

This project studies how false negatives affect contrastive retrieval. Standard InfoNCE assumes that every non-matching sample in a batch is a negative. However, in clustered or semantically overlapping datasets, two samples may not be exact pairs but may still be meaningfully similar. Pushing these samples apart too strongly can distort the learned representation space.

This repository compares standard InfoNCE with a false-negative-aware InfoNCE variant using synthetic paired data with known semantic cluster structure. The benchmark evaluates clean, clustered, and noisy-pair settings across different sample sizes. The results show that false-negative-aware contrastive learning can help when semantic overlap creates harmful negatives, but it is not a universal replacement for standard InfoNCE. Loss behavior depends on data structure, positive-pair quality, downweighting strength, and sample size.

---

## 1. Motivation

Contrastive learning is widely used for representation learning and retrieval. The core idea is simple: pull matched positive pairs together and push negative samples apart.

However, the negative-sampling assumption can be fragile. In many real datasets, especially multimodal, medical, and observational datasets, non-paired samples are not always true negatives. For example, two patients may not be the same exact pair, but they may share similar clinical patterns. Two images may not be matched, but they may represent similar semantic content.

If a contrastive loss treats these semantically related samples as ordinary negatives, the model may learn an embedding space that is too strict, fragmented, or semantically distorted.

This project studies that failure mode in a controlled setting.

---

## 2. Research Question

The main research question is:

> Can false-negative-aware InfoNCE improve retrieval when semantically similar samples are incorrectly treated as negatives?

The project tests this question by controlling:

- semantic cluster overlap
- clean versus noisy positive pairs
- sample size
- false-negative downweighting strength
- retrieval behavior under different data conditions

---

## 3. Background

### 3.1 Standard InfoNCE

Standard InfoNCE pulls each matched pair closer in embedding space while pushing all other batch samples away.

In a paired retrieval setting, each query has one known positive target. All other targets in the batch are treated as negatives.

This works well when most non-matching samples are genuinely unrelated.

### 3.2 False Negatives

A false negative occurs when a sample is treated as negative even though it is semantically similar to the query.

In this project, false negatives are simulated using known semantic cluster labels. Samples from the same cluster are not exact pairs, but they share latent semantic structure. Standard InfoNCE treats them as negatives. The false-negative-aware variant reduces the penalty for pushing away same-cluster non-pairs.

---

## 4. Method

The benchmark compares three loss settings:

| Loss Setting | Description |
|---|---|
| Standard InfoNCE | Treats every non-matching batch sample as a negative |
| FN-aware InfoNCE, alpha 0.5 | Moderately downweights same-cluster negatives |
| FN-aware InfoNCE, alpha 0.25 | Strongly downweights same-cluster negatives |

The false-negative-aware loss uses cluster labels only for controlled experimentation. The purpose is not to claim that cluster labels are always available in real datasets. Instead, the goal is to isolate and study how false-negative pressure affects contrastive retrieval behavior.

---

## 5. Dataset and Experimental Setup

The project uses synthetic paired feature vectors with known semantic cluster structure.

Each sample contains:

| Component | Description |
|---|---|
| Modality A | Synthetic feature vector for the query side |
| Modality B | Synthetic feature vector for the candidate side |
| Pair ID | Ground-truth exact match |
| Cluster label | Known semantic group used to identify possible false negatives |

Three dataset modes are tested:

| Dataset Mode | Purpose |
|---|---|
| Clean | Correct pairs with low false-negative pressure |
| Clustered | Many semantically similar non-pairs, increasing false-negative risk |
| Noisy | Some positive pair assignments are intentionally corrupted |

Two sample sizes are tested:

| Sample Size | Purpose |
|---|---|
| 5000 | Smaller controlled setting |
| 20000 | Larger setting with more retrieval candidates |

All main runs use 50 training epochs.

---

## 6. Experiment Matrix

The benchmark evaluates:

```text
3 dataset modes × 2 sample sizes × 3 loss settings = 18 runs
```

The goal is to test whether false-negative-aware contrastive learning moves retrieval behavior beyond standard InfoNCE under conditions where semantic overlap creates harmful negatives.

## 7. Evaluation Metrics

The project evaluates both retrieval quality and embedding behavior.

| Metric | Meaning |
|---|---|
| Recall@1 | Whether the correct match is ranked first |
| Recall@5 | Whether the correct match appears in the top 5 |
| Recall@10 | Whether the correct match appears in the top 10 |
| Recall@50 | Whether the correct match appears in the top 50 |
| Lift@K | Improvement over random retrieval |
| Positive similarity | Average similarity between true pairs |
| Same-cluster negative similarity | Similarity between semantically related non-pairs |
| Different-cluster negative similarity | Similarity between unrelated negatives |
| Training loss | Final optimization objective value |

This metric set is useful because retrieval metrics alone do not fully explain how the embedding space changes. Similarity diagnostics help show whether the model is separating true negatives while avoiding excessive punishment of semantically related samples.

## 8. Results

The results show that false-negative-aware InfoNCE is not universally better than standard InfoNCE.

Instead, its usefulness depends on the data condition.

### 8.1 Clean data

In clean settings, standard InfoNCE already performs strongly. False-negative-aware loss does not provide a clear improvement because there is little harmful false-negative pressure to correct.

This result is important because it shows that modifying the loss function is not automatically beneficial. If the standard objective already matches the data structure well, extra adjustments may add little value.

### 8.2 Clustered data

In clustered settings, false-negative-aware loss can improve some retrieval metrics, especially when semantically similar samples are frequently treated as negatives.

This supports the main hypothesis: downweighting same-cluster negatives can help when semantic overlap creates harmful contrastive pressure.

However, the improvement is not uniform across all metrics, sample sizes, or alpha values. This suggests that false-negative handling must be tuned carefully.

### 8.3 Noisy data

When positive pairs are corrupted, all loss settings degrade.

False-negative-aware loss may slightly improve some retrieval metrics, but it cannot fully repair wrong positive supervision. This shows that loss design cannot compensate for severely incorrect pair assignments.

The noisy setting connects this project to a broader lesson in multimodal learning: positive-pair quality and loss design must be studied together.

## 9. Key Findings

### 9.1 Standard InfoNCE is strong when data is clean

When positive pairs are correct and false-negative pressure is low, standard InfoNCE can already learn effective retrieval representations.

### 9.2 False-negative-aware loss helps mainly under semantic overlap

The false-negative-aware variant is most useful when many non-paired samples share semantic structure with the query.

### 9.3 Stronger downweighting is not always better

A smaller alpha reduces same-cluster negative pressure more strongly, but this does not guarantee better retrieval. Too much downweighting may weaken useful discrimination.

### 9.4 Loss design cannot fully fix corrupted positive pairs

When the positive-pair assignments are wrong, all methods suffer. This shows that false-negative handling is not a substitute for reliable positive supervision.

### 9.5 Evaluation must include similarity diagnostics

Recall@K shows retrieval quality, but similarity diagnostics reveal how the learned embedding space behaves. Both are needed to understand the effect of false-negative-aware training.

## 10. Limitations

This project uses synthetic data, so the results should be interpreted as controlled method-behavior analysis rather than direct evidence on real clinical or production datasets.

The false-negative-aware loss uses known cluster labels, which are available in the synthetic benchmark but may not be available in real-world settings. In practice, cluster membership may need to be estimated using metadata, labels, embeddings, or domain knowledge.

The benchmark also tests a limited number of alpha values. More values, repeated random seeds, harder cluster-overlap conditions, and real multimodal datasets would provide stronger evidence.

## 11. Future Work

Possible extensions include:

- testing more alpha values
- repeating experiments across multiple random seeds
- adding harder cluster-overlap settings
- testing estimated clusters instead of known synthetic clusters
- applying the method to authorized real paired datasets
- comparing with debiased contrastive learning variants
- analyzing embedding geometry after false-negative-aware training
- studying interaction between pair corruption and false-negative pressure

## 12. What I Learned

This project taught me that contrastive learning failure is not only about model architecture. It is also about how the loss function interprets the relationships between samples.

The most important lesson is:

> Not every non-matching sample is a true negative.

Standard InfoNCE can be very strong, but its assumptions should be tested. False-negative-aware training can help when semantic overlap is meaningful, but it must be evaluated together with data quality, positive-pair reliability, sample size, and embedding behavior.

This changed how I think about retrieval evaluation. A good benchmark should not only ask whether a loss improves Recall@K. It should also ask when it helps, when it fails, and what kind of embedding space it creates.
