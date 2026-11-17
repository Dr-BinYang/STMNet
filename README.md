# STMNet

**STMNet: A Spatio-Temporal Network for Time-Series Forecasting**

This repository provides the PyTorch implementation of STMNet, a deep learning framework for multivariate time-series forecasting.

## Dataset Preparation

Download the datasets from the following link:

[Dataset Download](https://figshare.com/s/d120b501a2143eee3cb8)

After downloading, place the dataset files in the `dataset/` directory.

The default configuration uses the Weather dataset:

```text
dataset/
└── weather/
    └── weather.csv
```

If you use a different dataset, update the corresponding dataset configuration in `run.py` or specify the appropriate command-line arguments.


## Overview

STMNet integrates multi-scale temporal modeling, spatio-temporal feature fusion, and an encoder–decoder architecture to capture complex patterns in time-series data.

The framework combines series decomposition with complementary feature extraction mechanisms to model temporal variations, feature-level spatial dependencies, and global representations.



## Repository Structure

```text
STMNet/
├── data_provider/
│   ├── data_factory.py
│   ├── data_loader.py
│   └── uea.py
│
├── dataset/
│   └── Dataset files
│
├── exp/
│   ├── exp_basic.py
│   └── exp_long_term_forecasting.py
│
├── model/
│   ├── attention.py
│   ├── decoder.py
│   ├── embed.py
│   ├── encoder.py
│   ├── MTM.py
│   ├── series_decomp.py
│   ├── STFus.py
│   └── STMNet.py
│
├── utils/
│   ├── data_analysis.py
│   ├── masking.py
│   ├── metrics.py
│   ├── plot_results.py
│   ├── save_loss_data.py
│   ├── timefeatures.py
│   └── tools.py
│
├── requirements.txt
└── run.py
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Dr-BinYang/STMNet.git
cd STMNet
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Environment Requirements**

The provided dependency configuration includes:

- Python
- PyTorch 2.6.0
- CUDA 12.6
- PyTorch Geometric 2.6.1

Make sure that the installed PyTorch, CUDA, and PyTorch Geometric packages are compatible with your environment. In particular, the PyTorch Geometric extension packages require matching precompiled wheels.

## Running Experiments

### 1. Default Configuration

Run the following command to start the training process:

```bash
python run.py
```

### 2. Custom Configuration

You can customize the experimental settings through command-line arguments.

For example:

```bash
python run.py \
    --data weather \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --seq_len 96 \
    --pred_len 192 \
    --batch_size 64 \
    --learning_rate 0.0001
```




