import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Set matplotlib backend to non-interactive mode for saving plots
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    """
    Adjust learning rate based on different scheduling strategies

    Args:
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        epoch: Current training epoch
        args: Configuration arguments
        printout: Whether to print learning rate updates
    """
    # Original commented learning rate schedule
    # lr = args.learning_rate * (0.2 ** (epoch // 2))

    # Define different learning rate adjustment strategies
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    # Apply learning rate adjustment if current epoch is in the schedule
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """
    Early stopping to prevent overfitting by monitoring validation loss
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            verbose: Whether to print stopping messages
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Call to check if training should stop early

        Args:
            val_loss: Current validation loss
            model: Model to save if improvement found
            path: Path to save model checkpoint
        """
        score = -val_loss

        if self.best_score is None:
            # First validation, save model
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # No improvement, increment counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found, reset counter and save model
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Save model checkpoint when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """Dictionary with dot notation access to attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    Standard scaler for data normalization using precomputed mean and std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        """Normalize data using stored mean and standard deviation"""
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Denormalize data back to original scale"""
        return (data * self.std) + self.mean


def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Save true values and predictions to CSV file

    Args:
        true: Ground truth values
        preds: Predicted values (optional)
        name: Output file path
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Visualize ground truth and predictions

    Args:
        true: Ground truth values
        preds: Predicted values (optional)
        name: Output plot file path
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Visualize attention weights or other weight matrices

    Args:
        weights: Weight matrix to visualize
        name: Output plot file path
    """
    fig, ax = plt.subplots()
    # Alternative colormap option: 'plasma_r'
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    """
    Adjust predictions based on ground truth anomaly states

    Args:
        gt: Ground truth labels
        pred: Predicted labels

    Returns:
        tuple: Adjusted ground truth and predictions
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Propagate anomaly detection backwards
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Propagate anomaly detection forwards
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """Calculate accuracy between predictions and ground truth"""
    return np.mean(y_pred == y_true)