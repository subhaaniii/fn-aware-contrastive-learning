# False-Negative-Aware Contrastive Learning: Controlled Experiment Summary

## 1. What method did I test?

This project compares standard symmetric InfoNCE with a false-negative-aware InfoNCE variant.

Standard InfoNCE treats every non-matching sample in a batch as a negative. In datasets with semantic overlap, this can create false negatives: samples that are not exact pairs but still belong to the same semantic group.

The false-negative-aware loss reduces the negative penalty for samples that share the same synthetic semantic cluster.

## 2. What dataset setup did I use?

This repository uses controlled synthetic paired data with two modalities:

- Modality A: synthetic feature vector
- Modality B: paired synthetic feature vector
- Cluster label: synthetic semantic group

Three dataset modes were tested:

| Mode | Description |
|---|---|
| Clean | Correct pairs with low false-negative pressure |
| Clustered | Semantically similar samples share cluster labels, increasing false-negative risk |
| Noisy | A fraction of positive pair assignments is intentionally corrupted |

The benchmark uses synthetic data so the false-negative structure, cluster overlap, pair corruption, and sample size can be explicitly controlled.

## 3. What metric did I measure?

The experiments report:

- Recall@1
- Recall@5
- Recall@10
- Recall@50
- Lift over random retrieval
- Positive-pair cosine similarity
- Same-cluster negative similarity
- Different-cluster negative similarity
- Training loss

## 4. What changed across experiments?

| Variable | Values tested |
|---|---|
| Loss function | Standard InfoNCE, FN-aware InfoNCE |
| FN-aware alpha | 0.5, 0.25 |
| Dataset mode | Clean, clustered, noisy |
| Sample size | 5000, 20000 |
| Training duration | 50 epochs |

## 5. Main Results

### 5000-sample benchmark

| Mode | Loss | Alpha | R@1 | R@5 | R@10 | R@50 | Lift@50 | Pos Sim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | Standard | - | 0.974 | 0.997 | 0.999 | 1.000 | 20.00x | 0.8047 |
| Clean | FN-aware | 0.5 | 0.972 | 0.997 | 0.999 | 1.000 | 20.00x | 0.8061 |
| Clean | FN-aware | 0.25 | 0.972 | 0.997 | 0.999 | 1.000 | 20.00x | 0.8068 |
| Clustered | Standard | - | 0.315 | 0.610 | 0.717 | 0.926 | 18.52x | 0.5621 |
| Clustered | FN-aware | 0.5 | 0.333 | 0.611 | 0.726 | 0.924 | 18.48x | 0.5643 |
| Clustered | FN-aware | 0.25 | 0.337 | 0.604 | 0.721 | 0.924 | 18.48x | 0.5652 |
| Noisy | Standard | - | 0.109 | 0.283 | 0.372 | 0.611 | 12.22x | 0.3400 |
| Noisy | FN-aware | 0.5 | 0.114 | 0.292 | 0.367 | 0.608 | 12.16x | 0.3426 |
| Noisy | FN-aware | 0.25 | 0.106 | 0.293 | 0.374 | 0.609 | 12.18x | 0.3430 |

### 20000-sample benchmark

| Mode | Loss | Alpha | R@1 | R@5 | R@10 | R@50 | Lift@50 | Pos Sim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | Standard | - | 0.9640 | 0.9973 | 0.9990 | 1.0000 | 80.00x | 0.8278 |
| Clean | FN-aware | 0.5 | 0.9638 | 0.9973 | 0.9990 | 0.9998 | 79.98x | 0.8280 |
| Clean | FN-aware | 0.25 | 0.9618 | 0.9970 | 0.9990 | 0.9998 | 79.98x | 0.8279 |
| Clustered | Standard | - | 0.2703 | 0.5178 | 0.6310 | 0.8523 | 68.18x | 0.6099 |
| Clustered | FN-aware | 0.5 | 0.2653 | 0.5143 | 0.6398 | 0.8570 | 68.56x | 0.6103 |
| Clustered | FN-aware | 0.25 | 0.2660 | 0.5155 | 0.6295 | 0.8543 | 68.34x | 0.6099 |
| Noisy | Standard | - | 0.1285 | 0.2835 | 0.3665 | 0.5625 | 45.00x | 0.3965 |
| Noisy | FN-aware | 0.5 | 0.1255 | 0.2810 | 0.3563 | 0.5660 | 45.28x | 0.3961 |
| Noisy | FN-aware | 0.25 | 0.1273 | 0.2825 | 0.3673 | 0.5633 | 45.06x | 0.3960 |

## 6. What did I learn?

### Clean mode

In clean data, standard InfoNCE already performed almost perfectly. FN-aware loss did not improve retrieval because there was little false-negative pressure to correct.

This is an important baseline: false-negative-aware learning should not be expected to improve every dataset.

### Clustered mode

In the 5000-sample clustered setup, FN-aware loss improved top-rank retrieval slightly. R@1 improved from 0.315 with standard InfoNCE to 0.333 with alpha 0.5 and 0.337 with alpha 0.25.

This supports the idea that false-negative-aware weighting can help when semantically similar samples are incorrectly treated as hard negatives.

At 20000 samples, the results were mixed. Standard InfoNCE had the best R@1 and R@5, while FN-aware alpha 0.5 had the best R@10 and R@50. This suggests the benefit depends on scale and the downweighting strength.

### Noisy mode

In noisy mode, all losses degraded because some positive pairs were intentionally corrupted.

FN-aware loss slightly improved some metrics, especially R@50 at 20000 samples, but it did not fully solve the problem. This shows that reducing false-negative pressure cannot completely compensate for wrong positive supervision.

### Alpha effect

The two FN-aware settings behaved differently:

- Alpha 0.5 applies moderate downweighting to same-cluster negatives.
- Alpha 0.25 applies stronger downweighting.

Stronger downweighting did not consistently improve retrieval. This suggests that too much relaxation of same-cluster negatives can reduce discrimination between nearby samples.

## Main conclusion

False-negative-aware contrastive learning is most useful when semantic overlap creates false negatives, but it is not a universal replacement for standard InfoNCE.

The main lesson is that loss design, cluster structure, positive-pair quality, and sample size interact strongly. A false-negative-aware objective can reduce harmful negative pressure, but it must be tuned carefully and cannot repair corrupted positive pairs by itself.

## Limitations

The benchmark uses synthetic paired features, so the false-negative structure and pair corruption rate are controlled by design. This makes the experiment useful for studying loss behavior, but it does not capture the full complexity of real multimodal datasets.

The results should not be interpreted as clinical or production-level retrieval performance. They show how standard InfoNCE and FN-aware InfoNCE behave under controlled clean, clustered, and noisy-pair conditions.

A stronger follow-up would include repeated seeds, additional alpha values, harder cluster-overlap settings, larger candidate pools, and evaluation on authorized real-world paired datasets.