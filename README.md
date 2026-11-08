# UvS-STG: Spatio-Temporal Graph Model Training and Unlearning

This repository implements training and unlearning workflows for spatio-temporal graph neural networks (STGNNs). It supports standard forecasting models such as STGCN, STGAT, and GWNet, and includes a machine-unlearning pipeline designed to remove the influence of selected faulty or forgotten nodes/subsets while preserving the remaining model behavior.

The project is organized around two main workflows:

- Training a graph-based forecasting model from scratch
- Applying unlearning to a pretrained model using retention, surrogate objectives, and EWC-style regularization

## Overview

Spatio-temporal forecasting models are commonly used for traffic, sensor, and graph-structured time series prediction. In real-world deployments, some nodes or time windows may become noisy, faulty, or require removal due to privacy or quality constraints. This repository addresses that need by combining:

- graph-based temporal prediction models,
- node/subset-based forgetting logic,
- population-aware Fisher information estimation,
- temporal generative replay,
- motif-based forgetting identification.

## Supported models

The codebase includes the following model families:

- STGCN: Spatio-Temporal Graph Convolutional Network
- STGAT: Spatio-Temporal Graph Attention Network
- GWNet: Graph WaveNet

## Main features

- Training pipeline for graph-based multivariate time series forecasting
- Node-level and subset-level unlearning
- Forgetting-set detection using motif-based criteria
- Retain / forget / surrogate loss decomposition
- Population-aware EWC (PA-EWC) regularization
- Temporal generative replay (T-GR) for surrogate data construction
- Optional CUDA support
- Flexible command-line configuration for experiments

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── train.py
├── unlearn.py
├── models/
│   ├── gwn.py
│   ├── stgat.py
│   └── stgcn.py
├── unlearning/
│   ├── motif_def.py
│   ├── pa_ewc.py
│   └── t_gr.py
└── utils/
    ├── VAR.py
    ├── data_loader.py
    ├── data_preprocess.py
    ├── filter_forget.py
    ├── replace_surrogate.py
    └── zscore_normalize.py
```

## Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The project expects a PyTorch environment and uses standard scientific Python libraries such as NumPy.

## Dataset format

This repository loads datasets from a directory containing:

- `train.npy`
- `test.npy`
- `adj_mx.pkl`

The train.npy and test.npy files initially have the shape (T, N, F), where:

- T: Number of time steps.
- N: Number of nodes in the graph.
- F: Number of features per node.

The adjacency matrix is expected to be a Python pickle object representing the graph connectivity matrix.

## Forget Set Structure
The forget set is stored in a JSON file that specifies the example faulty subset.

The JSON structure is organized as follows:

```json
{
    "0": [
        [0, 100]
    ]
}
```

Where:
- Key ("0"): Represents the node ID in the graph.
- Value: A list of time intervals to be forgotten for the corresponding node.
    - Each sublist: Contains two elements:
    - The first element represents the start index of the interval.
    - The second element represents the end index of the interval.

## Training

Use `train.py` to train a model on a dataset.

### Example

```bash
python train.py \
  --input /path/to/dataset \
  --model ./checkpoints \
  --type stgcn \
  --num-timesteps-input 12 \
  --num-timesteps-output 4 \
  --num-epochs 100 \
  --batch-size 128 \
  --learning-rate 1e-3
```

### Arguments

- `--input`: dataset directory
- `--model`: output directory for saved checkpoints
- `--type`: model type (`stgcn`, `stgat`, or `gwnet`)
- `--all`: train all supported models
- `--enable-cuda`: enable CUDA if available
- `--num-timesteps-input`: number of input time steps
- `--num-timesteps-output`: number of forecast steps
- `--num-epochs`: training epochs
- `--learning-rate`: optimizer learning rate
- `--batch-size`: training batch size

The training script saves model checkpoints into the directory specified by `--model`.

## Unlearning

Use `unlearn.py` to remove the influence of a specific node or a forget subset from a pretrained model.

### Node unlearning

```bash
python unlearn.py \
  --data-full /path/to/dataset \
  --model ./checkpoints \
  --type stgcn \
  --node-idx 10 \
  --unlearn-node \
  --threshold-node 0\
  --num-epochs 50 \
  --batch-size 128 \
  --learning-rate 1e-5 \
  --lambda-ewc 5.0 \
  --lambda-surrogate 1.0 \
  --lambda-retain 1.0 \
  --lambda-forget 1.0 \
  --forget-margin 0.1  
```

### Subset unlearning

```bash
python unlearn.py \
  --data-full /path/to/dataset \
  --forget-set /path/to/forget_example \
  --model ./checkpoints \
  --type stgcn \
  --threshold-subset 5.0 \
  --num-epochs 50 \
  --batch-size 128 \
  --learning-rate 1e-5 \
  --lambda-ewc 5.0 \
  --lambda-surrogate 1.0 \
  --lambda-retain 1.0 \
  --lambda-forget 1.0 \
  --forget-margin 0.1
```

### Key arguments

- `--data-full`: full training dataset directory
- `--data-standard`: optional standard dataset directory for evaluation
- `--forget-set`: file containing forget samples for subset unlearning
- `--node-idx`: node or nodes to forget
- `--unlearn-node`: enable node-level forgetting mode
- `--threshold-subset`: threshold used for subset-based forgetting detection
- `--threshold-node`: threshold used for node-level identification
- `--lambda-ewc`: weight for the EWC penalty
- `--lambda-surrogate`: weight for surrogate objective
- `--lambda-retain`: weight for retention objective
- `--lambda-forget`: weight for forgetting objective
- `--forget-margin`: target forgetting margin

## Training and unlearning workflow

The repository follows a typical graph unlearning workflow:

1. Load a graph-structured dataset and adjacency matrix.
2. Build forecasting samples with a fixed historical window and prediction horizon.
3. Train a spatio-temporal graph forecasting model.
4. Identify the forget set or faulty node.
5. Construct surrogate data and compare retain-versus-forget behavior.
6. Update the model with retain and surrogate losses while regularizing it using PA-EWC.
7. Save the unlearned checkpoint.

## Notes

- Training and unlearning logic are designed for graph datasets with one node representing a sensor or station.
- The code supports graph-wide and node-specific forgetting scenarios, making it useful for sensor repair, privacy removal, and robustness cleanup tasks.
- CUDA is optional; if unavailable, the code falls back to CPU execution.

## Acknowledgements

This repository implements spatio-temporal graph modeling and machine unlearning ideas combining graph neural forecasting architectures with regularized unlearning techniques.
