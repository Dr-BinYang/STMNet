# STMNet
STMNet for time series forecasting

## 📁 Project Overview




## 🎯 Core Module Descriptions

### 🧠 Model Architecture (`model/`)
- **`STMNet.py`** - Main model class with encoder-decoder architecture
- **`MTM.py`** - Multi-scale Temporal Memory for long-range dependencies
- **`STFus.py`** - Spatio-Temporal Fusion module
- **`attention.py`** - Various attention mechanisms (FullAttention, ProbAttention, etc.)

### 🗂️ Data Processing (`data_provider/`)
- **`data_factory.py`** - Factory pattern for dataset creation
- **`data_loader.py`** - PyTorch DataLoader implementations
- **`uea.py`** - Specialized handler for UEA time series datasets

### 🔬 Experiment Management (`exp/`)
- **`exp_basic.py`** - Base experiment class with training pipeline
- **`exp_long_term_forecasting.py`** - Long-term forecasting experiments

### 🛠️ Utility Library (`utils/`)
- **`metrics.py`** - Evaluation metrics (MSE, MAE, RMSE, MAPE, etc.)
- **`plot_results.py`** - Visualization tools for training curves and predictions
- **`timefeatures.py`** - Temporal feature encoding (hour, weekday, month, etc.)

## 🚀 Quick Start Guide

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Prepare Data**: Place datasets in the `dataset/` directory
3. **Run Experiments**: `python run.py --config your_config.yaml`
4. **Monitor Results**: Check `checkpoints/` for models, `results/` for visualizations

## 📊 Supported Features

- ✅ Multi-variate time series forecasting
- ✅ Long-term and short-term prediction
- ✅ Attention mechanism visualization
- ✅ Automatic model checkpointing
- ✅ Real-time training monitoring

## 🔧 Technical Stack

- **Framework**: PyTorch
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Time Series**: sktime

---

*Project follows modular design principles for extensibility and maintainability*
