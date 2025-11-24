# SLatK Ranking Loss - Recommendation System Ranking Loss Implementation

This project implements three ranking loss functions for recommendation systems: BPR Loss, Softmax Loss (SL), and SoftmaxLoss@K (SLatK). It supports multiple recommendation models (Matrix Factorization, LightGCN, XSimGCL) and various datasets (MovieLens 100K, Amazon, Gowalla, etc.).

## Table of Contents

- [How to Compile](#how-to-compile)
- [How to Execute](#how-to-execute)
- [Source File Descriptions](#source-file-descriptions)
- [Running Examples](#running-examples)
- [Tested Environments](#tested-environments)

## How to Compile

This project uses Python 3.9+ and requires no compilation. You only need to install the dependencies.

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation Steps

1. **Install PyTorch** (choose according to your CUDA version, or use CPU version):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

2. **Install other dependencies**:

```bash
pip install -r requirements.txt
```

3. **Prepare dataset** (MovieLens 100K):

```bash
python -m scripts.prepare_movielens
```

4. **Prepare original paper datasets** (Optional, Amazon/Gowalla):

```bash
# Download raw data from the following links:
# https://jmcauley.ucsd.edu/data/amazon/index_2014.html
# https://snap.stanford.edu/data/loc-Gowalla.html
# Prepare the raw data in the directory as follows:
# data
# ├── process_data.ipynb
# └── raw
#     ├── loc-gowalla_totalCheckins.txt.gz
#     ├── ratings_Books.csv
#     ├── ratings_Electronics.csv
#     └── ratings_Health_and_Personal_Care.csv
python data/process_data.py
```

## How to Execute

### Basic Training

Train model with default configuration:

```bash
python train.py
```

Train with custom configuration file:

```bash
python train.py --config cfgs/default.yaml
```

### Experiment Scripts

The project includes three experiment scripts for reproducing paper results:

1. **Figure 1 Experiment** (single gradient step effect comparison):

```bash
python figure1.py
```

2. **Figure 2 Experiment** (full training comparison of different loss functions):

```bash
python figure2.py
```

3. **Figure 3 Experiment** (matching between training K and evaluation K in SL@K):

```bash
python figure3.py
```

## Source File Descriptions

### Core Training Files

- **`train.py`**: Main training script supporting multiple models and loss functions, including complete training loop, evaluation, and wandb logging
- **`figure1.py`**: Experiment script 1, comparing the effects of different loss functions after a single gradient update step
- **`figure2.py`**: Experiment script 2, comparing the performance of BPR, SL, and SLatK loss functions during full training
- **`figure3.py`**: Experiment script 3, verifying the importance of matching training K value with evaluation K value in SL@K loss function

### Loss Function Module (`losses/`)

- **`losses/base.py`**: Base class `BaseRankingLoss` for loss functions, defining unified interface and Top-K quantile estimation methods
- **`losses/bpr.py`**: BPR Loss implementation, classic pairwise ranking loss
- **`losses/sl.py`**: Softmax Loss implementation, ranking loss based on softmax normalization
- **`losses/slatk.py`**: SoftmaxLoss@K (SLatK) implementation, adding Top-K weighting mechanism on top of Softmax Loss
- **`losses/__init__.py`**: Module export file
- **`losses/README.md`**: Detailed usage instructions for the loss function module

### Model Module (`models/`)

- **`models/base.py`**: Base class `BaseModel` for models, defining `full_item_scores()` and `l2_regularization()` interfaces
- **`models/mf.py`**: Matrix Factorization model implementation
- **`models/lightGCN.py`**: LightGCN graph neural network recommendation model implementation
- **`models/XSimGCL.py`**: XSimGCL contrastive learning recommendation model implementation
- **`models/__init__.py`**: Module export file

### Data Processing Module (`data/`)

- **`data/movielens.py`**: MovieLens 100K dataset loading and processing, including data reading, interaction construction, leave-one-out splitting, etc.
- **`data/read_proc_data.py`**: Interface for reading preprocessed datasets (Amazon, Gowalla, etc.)
- **`data/proc_*/`**: Preprocessed dataset directories containing training sets, test sets, and user/item summaries

### Evaluation Metrics Module (`metrics/`)

- **`metrics/ranking.py`**: Recommendation system evaluation metrics implementation, including Recall@K, NDCG@K, Precision@K
- **`metrics/__init__.py`**: Module export file

### Sampler Module (`samplers/`)

- **`samplers/neg.py`**: Negative sampler implementation, providing uniform negative sampling functionality
- **`samplers/__init__.py`**: Module export file

### Configuration Files (`cfgs/`)

- **`cfgs/default.yaml`**: Default training configuration file
- **`cfgs/figure1.yaml`**: Figure 1 experiment configuration file
- **`cfgs/figure2.yaml`**: Figure 2 experiment configuration file
- **`cfgs/figure3.yaml`**: Figure 3 experiment configuration file

### Utility Scripts (`scripts/`)

- **`scripts/prepare_movielens.py`**: MovieLens 100K dataset download and preprocessing script

### Test Files (`tests/`)

- **`tests/test_losses.py`**: Unit tests for loss functions, verifying Monte-Carlo β^K estimation and loss function correctness

### Other Files

- **`requirements.txt`**: Python dependency package list
- **`README.md`**: This file

## Running Examples

### Example 1: Training with Default Configuration

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Prepare MovieLens dataset (if not already downloaded)
python -m scripts.prepare_movielens

# 3. Train with default configuration
python train.py
```

### Example 2: Training with SLatK Loss Using Custom Configuration

Edit `cfgs/default.yaml` or create a new configuration file:

```yaml
dataset:
  type: proc
  root: data/proc_Amazon2014-Health
  batch_size: 1024

model:
  name: mf
  embedding_dim: 64
  user_reg: 1.0e-6
  item_reg: 1.0e-6

train:
  epochs: 50
  lr: 5.0e-3
  loss: slatk
  loss_params:
    tau_d: 0.5
    tau_w: 0.5
    topk: 10
  num_negatives: 200
  eval_k: 10
  seed: 42
```

Then run:

```bash
python train.py --config cfgs/default.yaml
```

### Example 3: Running Figure 2 Experiment (Loss Function Comparison)

```bash
python figure2.py
```

This script will:

1. Load MovieLens 100K dataset
2. Train MF models separately using BPR, SL, and SLatK loss functions
3. Output validation set Recall@K and NDCG@K metrics for each epoch
4. Compare the final performance of the three loss functions

### Example 4: Training with LightGCN Model

Modify the model settings in the configuration file:

```yaml
model:
  name: lightgcn
  embedding_dim: 64
  num_layers: 3
```

Then run training:

```bash
python train.py --config cfgs/default.yaml
```

## Tested Environments

This project has been tested on the following operating systems and environments:

- **macOS**: Darwin 25.2.0 (macOS Sequoia)
- **Linux**: Ubuntu 20.04+ (recommended)
- **Windows**: Windows 10/11 (via WSL or native Python)

### Python Versions

- Python 3.9+
- Python 3.10 (recommended)
- Python 3.11

### Dependency Versions

Main dependency package version requirements:

- torch >= 2.1.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0.1
- tqdm >= 4.66.0

### GPU Support

- CUDA 11.8+ (optional)
- CPU training supported (slower)

## Configuration Guide

Configuration files use YAML format and mainly include the following sections:

- **`dataset`**: Dataset configuration (type, path, batch size, etc.)
- **`model`**: Model configuration (model type, embedding dimension, regularization coefficients, etc.)
- **`train`**: Training configuration (loss function, learning rate, training epochs, evaluation K value, etc.)

For detailed configuration examples, please refer to the configuration files in the `cfgs/` directory.

## Notes

1. **Dataset Preparation**: Before first run, execute `python -m scripts.prepare_movielens` to download the MovieLens dataset
2. **GPU Memory**: When using LightGCN or XSimGCL models, if you encounter GPU memory issues, reduce `batch_size` or `num_negatives`
3. **Wandb Logging**: The training script uses wandb for experiment logging by default. You need to set the `WANDB_API_KEY` environment variable, or modify the code to disable wandb
4. **Random Seed**: All experiment scripts set random seeds to ensure reproducibility

## License

This project is for academic research use only.

## Contact

For questions or suggestions, please submit an Issue or contact the project maintainer.
