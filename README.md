# ExperimentOneData

Experimental data for the paper: *[TODO: paper title and link]*.

This repository contains the complete training output -- per-epoch metrics, model checkpoints, and configuration snapshots -- for all four data ordering strategies studied in the paper. For the training framework, experiment code, analysis tools, and reproduction instructions, see the companion code repository: [OrderedLearning](https://github.com/JordanRL/OrderedLearning).

## Experiment

A 2-layer transformer (7.7M parameters) trained on modular addition `(a + b) mod 9973` with 300K training pairs and 1M test pairs, using four data ordering strategies under identical hyperparameters (seed 199, AdamW, lr=0.001, weight_decay=0.1, batch_size=256). Training ran for up to 5,000 epochs with early stopping at 99.5% validation accuracy.

All runs used full instrumentation (19 hooks) recording 60+ metrics per epoch.

| Strategy | Epochs | Val Accuracy | Early Stopped | Duration |
|---|---|---|---|---|
| **stride** | 487 | 99.56% | Yes | 2.6 hrs |
| **fixed-random** | 659 | 99.63% | Yes | 3.2 hrs |
| **random** | 4,999 | 0.30% | No | 25.9 hrs |
| **target** | 4,999 | 0.01% | No | 25.9 hrs |

Hardware: NVIDIA RTX 4090, CUDA 12.4, PyTorch 2.4.1, Python 3.11.10.

## Repository Structure

```
├── stride/                     # Stride ordering strategy
├── target/                     # Target-sorted ordering strategy
├── random/                     # Reshuffled-each-epoch random ordering
├── fixed-random/               # Fixed random ordering (shuffled once)
└── README.md
```

Each strategy directory contains:

| File | Format | Description |
|---|---|---|
| `experiment_config.json` | JSON | Full hyperparameters and environment metadata |
| `summary.json` | JSON | Training outcome: timing, final metrics, early stopping |
| `{strategy}.jsonl` | JSONL (LFS) | Per-epoch metrics from all 19 instrumentation hooks |
| `checkpoints/` | PyTorch (LFS) | Model + optimizer state every 50 epochs |

## Data Sizes

All `.pt` and `.jsonl` files are tracked with [Git LFS](https://git-lfs.github.com/). You must have Git LFS installed to clone this repository.

| Strategy | JSONL | Checkpoints | Total |
|---|---|---|---|
| stride | 23 MB | 11 files (93 MB each) | ~1 GB |
| fixed-random | 31 MB | 15 files (93 MB each) | ~1.3 GB |
| random | 233 MB | 101 files (93 MB each) | ~8.8 GB |
| target | 232 MB | 101 files (93 MB each) | ~8.8 GB |

Total repository size: approximately 20 GB.

## JSONL Metrics Format

Each line in the JSONL files is a JSON object representing one epoch's metrics, namespaced by hook:

```
training_metrics/       Loss, accuracy, learning rate, perplexity
norms/                  Layer-wise gradient norms
fourier/                Spectral entropy, peak frequency, significant frequencies
hessian/                Hessian-based loss landscape metrics
attention/              Attention head statistics
counterfactual/         Gradient decomposition (ordering vs target components)
adam_dynamics/          Optimizer momentum and update analysis
...
```

For complete metric documentation, see the [hook reference](https://github.com/JordanRL/OrderedLearning/tree/master/docs) in the code repository.

## Checkpoint Format

Each `.pt` file is a dictionary saved with `torch.save()` containing model weights, optimizer state, scheduler state, and RNG states sufficient to resume training from that epoch.

**Security note:** Checkpoint files are loaded with `weights_only=False` and can execute arbitrary code. Only load checkpoints from sources you trust. See the [PyTorch serialization docs](https://pytorch.org/docs/stable/notes/serialization.html) for details.

## Working With This Data

### Using the analysis tools (recommended)

Clone both repositories and point the analysis tools at this data:

```bash
git clone https://github.com/JordanRL/OrderedLearning.git
git clone https://github.com/JordanRL/ExperimentOneData.git

cd OrderedLearning
pip install -r requirements.txt

# Plot validation accuracy across strategies
python analyze_experiment.py mod_arithmetic compare \
    --metrics training_metrics/validation_accuracy \
    --output-dir ../ExperimentOneData \
    --smooth 0.9

# Export a comparison table
python analyze_experiment.py mod_arithmetic export_table \
    --metrics training_metrics/validation_accuracy \
    --output-dir ../ExperimentOneData \
    --format latex
```

### Loading data directly

The JSONL and JSON files can be read with Python's standard library:

```python
import json

# Load experiment config
with open("stride/experiment_config.json") as f:
    config = json.load(f)

# Load training summary
with open("stride/summary.json") as f:
    summary = json.load(f)

# Stream per-epoch metrics
with open("stride/stride.jsonl") as f:
    for line in f:
        epoch = json.load(line)
        print(epoch["training_metrics/validation_accuracy"])
```

Checkpoints require PyTorch:

```python
import torch

checkpoint = torch.load("stride/checkpoints/checkpoint_200.pt", weights_only=False)
```

## Reproducing From Scratch

To regenerate this data from the code repository, see the [reproduction instructions](https://github.com/JordanRL/OrderedLearning#replicating-the-paper-experiment) in the OrderedLearning README.

## License

MIT -- see [LICENSE](LICENSE).
