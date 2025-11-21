# STMNet
STMNet for time series forecasting

## 📁 Project Overview






## 🚀 Quick Start Guide

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Prepare Data**: Place datasets in the `dataset/` directory
3. **Run Experiments**: `python run.py`
4. **Monitor Results**: Check `checkpoints/` for models, `results/` for visualizations


## 🎯 Core Module Descriptions

### 🗂️ Data Processing (`data_provider/`)
- **`data_factory.py`** - Factory pattern for dataset creation
- **`data_loader.py`** - PyTorch DataLoader implementations


### 🛠️ Utility Library (`utils/`)
- **`metrics.py`** - Evaluation metrics (MSE, MAE, RMSE, MAPE, etc.)
- **`plot_results.py`** - Visualization tools for training curves and predictions
- **`timefeatures.py`** - Temporal feature encoding (hour, weekday, month, etc.)


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
