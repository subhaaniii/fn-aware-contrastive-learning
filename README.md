# False-Negative-Aware Contrastive Learning

A controlled experiment on how false negatives affect contrastive retrieval.

This repository compares **standard InfoNCE** with a **false-negative-aware InfoNCE variant** using synthetic paired data with known semantic cluster structure. The goal is to study loss-function behavior under clean, clustered, and noisy-pair conditions.

## Research Report

A paper-style summary of this project is available here:

[Read the paper-style report](docs/paper_style_report.md)

---

## Problem

Contrastive learning usually assumes that every non-matching sample in a batch is a negative.

That assumption is not always safe.

In real multimodal datasets, two samples may not be exact pairs, but they can still be semantically similar. If contrastive learning pushes those samples apart too strongly, the model may learn a representation that is overly strict or semantically distorted.

This repository studies that problem in a controlled setup.

## Core Question

`Can false-negative-aware contrastive learning improve retrieval when some “negative” samples are actually semantically related to the query?`

This project studies that question by varying:

- semantic cluster overlap
- noisy positive pair assignments
- dataset size
- false-negative downweighting strength

---

## Hypothesis

False-negative-aware contrastive learning should help most when the dataset contains many semantically similar non-paired samples.

However, the benefit may disappear when:

- the data is already clean
- positive pairs are corrupted
- same-cluster negatives are downweighted too strongly
- standard InfoNCE is already sufficient

---

## Methods

### Standard InfoNCE

Standard InfoNCE pulls matched pairs together and pushes all other batch samples apart.

In this setup, every non-matching sample is treated as a negative.

### False-Negative-Aware InfoNCE

The false-negative-aware variant uses synthetic semantic cluster labels.

If two samples are not exact pairs but belong to the same cluster, their negative contribution is reduced.

Two downweighting strengths are tested:

| Loss | Meaning |
|---|---|
| FN-aware alpha 0.5 | Moderate downweighting of same-cluster negatives |
| FN-aware alpha 0.25 | Stronger downweighting of same-cluster negatives |

---

## Related Work

This project is inspired by contrastive representation learning, false-negative handling in contrastive objectives, and supervised/debiased contrastive learning.

For the full related-work discussion and references, see the [paper-style report](docs/paper_style_report.md).

---

## Metrics

The benchmark evaluates retrieval quality and embedding behavior using Recall@K, Lift@K, positive-pair similarity, same-cluster negative similarity, different-cluster negative similarity, and training loss.

For full metric definitions, see the [paper-style report](docs/paper_style_report.md).

## Experiment Matrix

| Variable | Values |
|---|---|
| Loss function | Standard InfoNCE, FN-aware InfoNCE |
| Alpha | 0.5, 0.25 |
| Dataset mode | Clean, clustered, noisy |
| Sample size | 5000, 20000 |
| Epochs | 50 |

Total benchmark runs:

```text
3 dataset modes × 2 sample sizes × 3 loss settings = 18 runs
```
---

## False-Negative Visualization

The plots below summarize the synthetic feature space and the main false-negative-aware comparison.

<table>
  <tr>
    <th>Input feature space</th>
    <th>Recall comparison</th>
    <th>Similarity diagnostic</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="figures/fn_aware_input_feature_space.png">
        <img src="figures/fn_aware_input_feature_space.png" alt="Input feature space" width="100%">
      </a>
    </td>
    <td width="33%">
      <a href="figures/fn_aware_recall_comparison.png">
        <img src="figures/fn_aware_recall_comparison.png" alt="Recall comparison" width="100%">
      </a>
    </td>
    <td width="33%">
      <a href="figures/fn_aware_similarity_diagnostic.png">
        <img src="figures/fn_aware_similarity_diagnostic.png" alt="Similarity diagnostic" width="100%">
      </a>
    </td>
  </tr>
</table>

Each panel links to the full-resolution figure.

| Panel | What to notice |
|---|---|
| **Input feature space** | Circles and crosses represent the two paired modalities. Colors indicate synthetic semantic clusters. Same-colored non-paired samples are potential false negatives under standard InfoNCE. |
| **Recall comparison** | Standard InfoNCE and FN-aware variants are compared across clean, clustered, and noisy settings. This shows whether downweighting same-cluster negatives improves retrieval. |
| **Similarity diagnostic** | Positive-pair similarity is compared with same-cluster negative similarity and different-cluster negative similarity. This helps show whether the model separates true negatives while avoiding excessive punishment of semantically related samples. |

These figures are qualitative diagnostics. The main conclusions are based on the quantitative retrieval results in `experiments/results_table.csv`.

---

## Main Findings

The results show that false-negative-aware InfoNCE is **not universally better** than standard InfoNCE.

Instead, its usefulness depends on the data condition.

### Clean data

Standard InfoNCE already performs almost perfectly. FN-aware loss does not improve retrieval because there is little false-negative pressure to fix.

### Clustered data

FN-aware loss improves some top-rank retrieval metrics in the 5000-sample clustered setup. This supports the idea that downweighting same-cluster negatives can help when semantically similar samples are incorrectly treated as hard negatives.

At 20000 samples, the results are mixed. Standard InfoNCE remains strong, while FN-aware loss improves some mid/top-k metrics depending on alpha.

### Noisy data

When positive pairs are corrupted, all losses degrade.

FN-aware loss slightly improves some retrieval metrics, but it cannot fully repair wrong positive supervision.

---

## Key Takeaway

False-negative-aware contrastive learning can help when semantic overlap creates harmful negatives, but it is not a universal replacement for standard InfoNCE.

The main lesson is:

> Loss design, semantic cluster structure, positive-pair quality, alpha strength, and sample size must be evaluated together.

---

## Repository Structure

```text
fn-aware-contrastive-learning/
│
├── src/
│   ├── make_demo_data.py
│   ├── train.py
│   ├── losses.py
│   ├── model.py
│   ├── metrics.py
│   └── collect_results.py
│
├── data_demo/
│   ├── demo_pairs.csv
│   └── demo_metadata.json
│
├── experiments/
│   ├── results_table.csv
│   └── results_summary.md
│
├── docs/
│   ├── method_overview.md
│   ├── experiment_design.md
│   ├── loss_explanation.md
│   ├── metric_explanation.md
│   └── reproducibility_notes.md
│
├── requirements.txt
└── README.md
```

---

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate clustered synthetic data:

```powershell
python src/make_demo_data.py --mode clustered --n-samples 5000 --n-clusters 50
```

Train standard InfoNCE:

```powershell
python src/train.py --loss standard --epochs 50 --batch-size 512 --output-dir outputs/clustered_standard --checkpoint-dir checkpoints/clustered_standard
```

Train false-negative-aware InfoNCE:

```powershell
python src/train.py --loss fn_aware --alpha 0.5 --epochs 50 --batch-size 512 --output-dir outputs/clustered_fn_aware --checkpoint-dir checkpoints/clustered_fn_aware
```

Collect final metrics:

```powershell
python src/collect_results.py
```

---

## Results

- [Full controlled experiment summary](experiments/results_summary.md)
- [Result table](experiments/results_table.csv)

---

## Documentation

- [Method overview](docs/method_overview.md)
- [Experiment design](docs/experiment_design.md)
- [Loss explanation](docs/loss_explanation.md)
- [Metric explanation](docs/metric_explanation.md)
- [Reproducibility notes](docs/reproducibility_notes.md)

---

## References

References are included in the [paper-style report](docs/paper_style_report.md).
