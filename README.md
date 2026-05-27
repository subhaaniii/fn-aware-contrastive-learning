# False-Negative-Aware Contrastive Learning

A controlled method-study comparing standard InfoNCE with false-negative-aware InfoNCE for multimodal retrieval.

The project studies a common issue in contrastive learning: not every non-matching sample is truly negative. In datasets with semantic overlap, two samples may not be exact pairs but may still belong to the same semantic group. Treating those samples as hard negatives can create false-negative pressure.

## Research Question

Can false-negative-aware contrastive learning improve retrieval when semantically similar samples are incorrectly treated as negatives?

## What This Repository Shows

This repository demonstrates:

- how standard InfoNCE behaves under clean, clustered, and noisy-pair settings
- how false-negative-aware InfoNCE changes retrieval behavior
- how semantic cluster labels can be used to downweight likely false negatives
- how retrieval metrics change when sample size increases from 5000 to 20000
- why loss design cannot fully compensate for corrupted positive pairs

## Methods Compared

| Method | Description |
|---|---|
| Standard InfoNCE | Treats all non-matching samples in a batch as negatives |
| FN-aware InfoNCE | Downweights non-matching samples that share the same semantic cluster |
| FN-aware alpha 0.5 | Moderate downweighting of same-cluster negatives |
| FN-aware alpha 0.25 | Stronger downweighting of same-cluster negatives |

## Dataset Setup

This repository uses controlled synthetic paired data with two modalities:

- Modality A: synthetic feature vector
- Modality B: paired synthetic feature vector
- Cluster label: synthetic semantic group

Three dataset modes are included:

| Mode | Purpose |
|---|---|
| Clean | Tests behavior when pairings are correct and false-negative pressure is low |
| Clustered | Tests behavior when many samples are semantically similar, increasing false-negative risk |
| Noisy | Tests behavior when some positive pair assignments are intentionally corrupted |

The benchmark uses synthetic paired features so that false-negative structure, cluster overlap, pair corruption, and sample size can be explicitly controlled.

## Experiments

The final benchmark compares:

- loss function: standard InfoNCE vs FN-aware InfoNCE
- FN-aware alpha: 0.5 and 0.25
- dataset mode: clean, clustered, noisy
- sample size: 5000 and 20000
- training duration: 50 epochs

Metrics reported:

- Recall@1
- Recall@5
- Recall@10
- Recall@50
- lift over random retrieval
- positive-pair cosine similarity
- same-cluster negative similarity
- different-cluster negative similarity
- training loss

## Key Findings

False-negative-aware InfoNCE did not universally outperform standard InfoNCE. Its effect depended on the dataset condition.

Main observations:

- In clean data, standard InfoNCE already performed almost perfectly.
- In clustered data, FN-aware loss improved some top-rank retrieval metrics, especially in the 5000-sample setup.
- In noisy data, all methods degraded because some positive pairs were intentionally corrupted.
- FN-aware loss can reduce false-negative pressure, but it cannot fully repair wrong positive supervision.
- Stronger downweighting was not always better, showing that alpha needs tuning.

The main lesson is that loss design, semantic overlap, positive-pair quality, and sample size interact strongly.

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

Collect results:

```powershell
python src/collect_results.py
```

## Documentation

- [Method overview](docs/method_overview.md)
- [Experiment design](docs/experiment_design.md)
- [Loss explanation](docs/loss_explanation.md)
- [Metric explanation](docs/metric_explanation.md)
- [Reproducibility notes](docs/reproducibility_notes.md)
- [Controlled experiment results](experiments/results_summary.md)

## Limitations

This repository is a controlled method-analysis project. It uses synthetic paired features and semantic cluster labels so that false-negative structure can be explicitly studied.

The results should be interpreted as evidence of loss-function behavior and retrieval-evaluation reasoning, not as a claim of performance on real clinical data or production retrieval systems.

A stronger follow-up would include repeated random seeds, additional alpha values, harder cluster-overlap settings, and evaluation on authorized real-world paired datasets.