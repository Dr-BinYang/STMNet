# STMNet
STMNet for time series forecasting

## 📁 Project Overview

STMNet/ # 🏠 Root Directory

├── 📁 checkpoints/ # 💾 Model Checkpoints Directory

│ └── (Saved model weights and training checkpoints)

├── 📁 data_provider/ # 🗂️ Data Loading Modules

│ ├── 🐍 data_factory.py # Data factory for dataset creation

│ ├── 🐍 data_loader.py # Data loader with batching

│ └── 🐍 uea.py # UEA dataset handler

├── 📁 dataset/ # 📊 Dataset Storage

│ └── (Raw and processed data files)

├── 📁 exp/ # 🔬 Experiment Configuration

│ ├── 🐍 exp_basic.py # Base experiment class

│ └── 🐍 exp_long_term_forecasting.py # Long-term forecasting experiments

├── 📁 model/ # 🧠 Core Model Architecture

│ ├── 🐍 init.py # Package initialization

│ ├── 🐍 attention.py # Attention mechanisms

│ ├── 🐍 decoder.py # Decoder module

│ ├── 🐍 embed.py # Embedding layers

│ ├── 🐍 encoder.py # Encoder module

│ ├── 🐍 MTM.py # Multi-scale Temporal Memory

│ ├── 🐍 series_decomp.py # Series decomposition

│ ├── 🐍 STFus.py # Spatio-Temporal Fusion

│ └── 🐍 STMNet.py # 🌟 Main model definition

├── 📁 utils/ # 🛠️ Utility Functions

│ ├── 🐍 init.py # Package initialization

│ ├── 🐍 data_analysis.py # Data analysis tools

│ ├── 🐍 masking.py # Masking utilities

│ ├── 🐍 metrics.py # 📈 Evaluation metrics

│ ├── 🐍 plot_results.py # 📊 Visualization tools

│ ├── 🐍 save_loss_data.py # Loss data serialization

│ ├── 🐍 timefeatures.py # ⏰ Time feature extraction

│ └── 🐍 tools.py # General utilities

├── 📄 requirements.txt # 📋 Python dependencies

└── 📄 run.py # 🚀 Main execution script








## 🚀 Quick Start Guide

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Prepare Data**: Place datasets in the `dataset/` directory
3. **Run Experiments**: `python run.py`
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
