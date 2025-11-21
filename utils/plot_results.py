from torch.optim import lr_scheduler
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import shutil
import time
import warnings
import numpy as np
import os
from exp.exp_basic import Exp_Basic
from torchsummary import summary
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


def plot_results(batch_y, preds, epoch_index, iter_count, task, phase='train'):
    """
    Plot and save comparison charts between predictions and ground truth

    Args:
        batch_y: Ground truth values [B, L, N]
        preds: Predicted values [B, L, N]
        epoch_index: Current epoch number
        iter_count: Current iteration count
        task: Task name (used for building save path)
        phase: Phase identifier ('train', 'val', 'test')

    Directory structure:
    ./results/
    └── 96_96/           # Task name
        ├── epoch_1/     # Epoch 1
        │   └── train/   # Training phase
        │       ├── sample_0.png  # Sample 0 from batch 0
        │       ├── sample_1.png  # Sample 1 from batch 0
        │       └── sample_5.png  # Last sample from batch 1 (assuming batch_size=5)
        ├── epoch_2/     # Epoch 2
        │   └── train/
        │       ├── sample_0.png  # Restart numbering from 0
        │       └── sample_4.png  # Last sample
        └── epoch_3/
            ├── train/
            └── val/     # Validation phase results
    """
    # Ensure data is on CPU and convert to numpy
    batch_y = batch_y.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()

    batch_size = batch_y.shape[0]
    seq_len = batch_y.shape[1]

    # Build save path: ./results/{task}/epoch_{epoch}/{phase}/
    dir_path = os.path.join('./results',
                            str(task),
                            f'epoch_{epoch_index}',
                            phase)
    os.makedirs(dir_path, exist_ok=True)

    # Calculate starting index for samples in this batch
    start_index = (iter_count - 1) * batch_size  # iter_count starts from 1

    # Plot image for each sample
    for i in range(batch_size):
        # Absolute sample number within the epoch
        sample_number = start_index + i

        # Create new figure
        plt.figure(figsize=(10, 6), dpi=400)

        # Extract true and predicted values for the i-th sample (take first feature)
        fea = 2  # Output feature dimension
        true = batch_y[i, :, fea]  # [L]
        pred = preds[i, :, fea]  # [L]

        # Plot curves
        plt.plot(range(seq_len), true, label='GroundTruth')
        plt.plot(range(seq_len), pred, label='Prediction')

        # Add annotations
        plt.title(f"Epoch {epoch_index} | {phase.capitalize()} Sample {sample_number}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(False)

        # Save image
        save_path = os.path.join(dir_path, f'sample_{sample_number}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    return 0


def plot_hierarchy_prediction(hierarchy_prediction, batch_y, epoch_index):
    """
    Visualize hierarchical prediction results

    Args:
        hierarchy_prediction: Tensor - shape [levels, batch_size, seq_len, features] (4, 32, 192, 21)
        batch_y: Tensor - Ground truth values, shape [batch_size, seq_len, features] (32, 192, 21)
        epoch_index: int - Current epoch index
    """
    # Ensure data is on CPU and convert to numpy
    batch_y = batch_y.detach().cpu().numpy()
    hierarchy_prediction = hierarchy_prediction.detach().cpu().numpy()

    # Dimension validation
    assert hierarchy_prediction.ndim == 4, "Input dimension should be [levels, batch, seq, features]"
    levels, batch_size, seq_len, num_features = hierarchy_prediction.shape
    assert batch_y.shape == (batch_size, seq_len, num_features), "Ground truth shape mismatch"

    # Generate default batch names
    batch_names = [f'sample_{i:03d}' for i in range(batch_size)]

    # Build save path: ./results/{task}/epoch_{epoch}/{phase}/
    save_dir = os.path.join('./results',
                            str('hierarchy_prediction'),
                            f'epoch_{epoch_index}')
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through each sample
    for batch_idx in range(batch_size):
        # Create 21x4 subplot layout
        fig, axs = plt.subplots(
            nrows=num_features,
            ncols=levels,
            figsize=(24, 5 * num_features),  # Dynamic height adjustment
            squeeze=False,
            sharex=True
        )

        # Iterate through each feature dimension
        for feature_idx in range(num_features):
            # Get ground truth sequence for current feature
            y_true = batch_y[batch_idx, :, feature_idx]  # [seq_len]

            # Iterate through each hierarchy level
            for level_idx in range(levels):
                # Get prediction sequence for current level
                y_pred = hierarchy_prediction[level_idx, batch_idx, :, feature_idx]  # [seq_len]

                # Select corresponding subplot
                ax = axs[feature_idx, level_idx]

                # Plot curves
                ax.plot(y_true, label='True')
                ax.plot(y_pred, label='Pred')

                # Set subplot title (only for first row)
                if feature_idx == 0:
                    ax.set_title(f'Level {level_idx + 1}', pad=20)

                # Set row labels (only for first column)
                if level_idx == 0:
                    ax.set_ylabel(f'Feature {feature_idx + 1}', rotation=0, ha='right', va='center')

                # Show legend (only for last feature and last level)
                if feature_idx == num_features - 1 and level_idx == levels - 1:
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, -0.2))

        # Adjust layout
        plt.tight_layout()

        # Save image
        save_path = os.path.join(save_dir, f'{batch_names[batch_idx]}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()


def plot_attention_heatmap(weights, save_dir="./results/attention_heatmap"):
    """
    Plot and save Gaussian-blurred multi-head attention heatmaps (8 heads)

    Args:
        weights: Attention weight tensor, shape [B, H, L, L]
        save_dir: Save directory path, automatically created
    """
    import os
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt

    # Create directory
    batch_idx = 0
    weights = weights[batch_idx].detach().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamp filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(save_dir, f"attention_gaussian_{timestamp}.png")

    # Adjust layout parameters
    fig, axes = plt.subplots(2, 4, figsize=(24, 12),
                             gridspec_kw={'wspace': 0.2, 'hspace': 0.3},
                             subplot_kw={'xticks': [], 'yticks': []})
    axes = axes.flatten()

    # Unified Gaussian interpolation parameters
    interp_method = 'gaussian'
    sigma = 2.0  # Gaussian blur intensity

    # Plot each head
    for i in range(8):
        # Use imshow instead of heatmap
        im = axes[i].imshow(
            weights[i],
            cmap="viridis",
            interpolation=interp_method,
            aspect='auto',  # Automatic aspect ratio adjustment
            filterrad=sigma,  # Gaussian kernel radius
            vmin=np.min(weights),  # Unified color range
            vmax=np.max(weights)
        )
        axes[i].set_title(f"Head {i + 1}\nGaussian(σ={sigma})", fontsize=12)

    # Add global color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # Save output
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()