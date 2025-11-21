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
from model.STMNet import Model
from torchsummary import summary
import matplotlib.pyplot as plt
from utils.plot_results import plot_results, plot_hierarchy_prediction
from utils.save_loss_data import save_loss_data


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    Experiment class for long-term time series forecasting
    Handles training, validation, testing, and model management
    """

    def __init__(self, args):
        """
        Initialize long-term forecasting experiment

        Args:
            args: Configuration arguments for the experiment
        """
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """Build and return the STMNet model"""
        model = Model(self.args)
        return model

    def _get_data(self, flag):
        """
        Get dataset and data loader for specified split

        Args:
            flag: Data split identifier ('train', 'val', 'test')

        Returns:
            tuple: (dataset, data_loader)
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """Select Adam optimizer with configured learning rate"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Select MSE loss as primary criterion"""
        criterion = nn.MSELoss()
        return criterion

    def train(self):
        """
        Main training loop with validation, testing, and early stopping
        Includes learning rate scheduling and model checkpointing
        """
        # Initialize data loaders
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Initialize loss tracking
        mse_loss_records = {'train': [], 'valid': [], 'test': []}

        # Create checkpoints directory
        os.makedirs(self.args.checkpoints, exist_ok=True)
        best_model_path = os.path.join(self.args.checkpoints, 'checkpoint.pth')

        time_now = time.time()
        train_steps = len(train_loader)

        # Initialize optimizer and loss functions
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  # Primary loss: MSE
        mae_criterion = nn.L1Loss()  # Secondary loss: MAE

        # Learning rate scheduler with plateau detection
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=model_optim,
            mode='min',  # Minimize validation loss
            factor=0.5,  # Halve learning rate on plateau
            patience=self.args.patience,  # Epochs to wait before reducing LR
            verbose=True,  # Print LR changes
            threshold=1e-4,  # Minimum improvement threshold
            min_lr=1e-7  # Minimum learning rate
        )

        # Early stopping initialization
        early_stop_counter = 0
        best_vali_loss = float('inf')  # Track best validation loss

        # Training epochs loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_MSE_loss = []
            train_MAE_loss = []

            self.model.train()
            epoch_time = time.time()

            # Batch training loop
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # Move data to appropriate device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Forward pass
                outputs, hierarchy_prediction = self.model(batch_x, batch_x_mark)

                # Calculate losses
                mse_loss = criterion(outputs, batch_y)
                mae_loss = mae_criterion(outputs, batch_y)

                # Track losses
                train_MSE_loss.append(mse_loss.item())
                train_MAE_loss.append(mae_loss.item())

                # Backward pass and optimization
                mse_loss.backward()
                model_optim.step()

                # Visualization during training
                if self.args.plot_result == True and epoch > 3 and batch_y.shape[0] * (iter_count - 1) < 500:
                    plot_results(
                        batch_y=batch_y,
                        preds=outputs,
                        epoch_index=epoch + 1,
                        iter_count=iter_count,
                        task=f"{self.args.seq_len}_{self.args.pred_len}",
                        phase='train'
                    )

                # Hierarchy prediction visualization
                if self.args.plot_hierarchy_prediction == True and i == 0:
                    plot_hierarchy_prediction(hierarchy_prediction, batch_y, epoch)

            # Calculate epoch time
            elapsed_time = time.time() - epoch_time
            if elapsed_time < 60:
                print("Cost time: {:>8.3f} seconds".format(elapsed_time))
            else:
                elapsed_minutes = elapsed_time / 60
                print("Cost time: {:>8.3f} minutes".format(elapsed_minutes))

            # Calculate average losses for the epoch
            train_MSE_loss, train_MAE_loss = np.average(train_MSE_loss), np.average(train_MAE_loss)
            vali_MSE_loss, vali_MAE_loss = self.vali(vali_data, vali_loader)
            test_MSE_loss, test_MAE_loss = self.test(test_data, test_loader, epoch)

            # Print epoch summary
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train MSE Loss: {train_MSE_loss:.4f}  Train MAE Loss: {train_MAE_loss:.4f} | "
                  f"Vali MSE Loss: {vali_MSE_loss:.4f}  Vali MAE Loss: {vali_MAE_loss:.4f} | "
                  f"Test MSE Loss: {test_MSE_loss:.4f}  Test MAE Loss: {test_MAE_loss:.4f}")

            # Learning rate adjustment
            old_lr = model_optim.param_groups[0]["lr"]
            scheduler.step(vali_MSE_loss)  # Update based on validation loss
            new_lr = model_optim.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"⚠ Learning rate adjusted from {old_lr:.7f} to {new_lr:.7f}")

            # Model checkpointing and early stopping
            if vali_MSE_loss < best_vali_loss:
                best_vali_loss = vali_MSE_loss
                torch.save(self.model.state_dict(), best_model_path)
                early_stop_counter = 0  # Reset early stopping counter
                print(f"✓ Better model found, validation loss: {vali_MSE_loss:.4f}")
            else:
                early_stop_counter += 1
                print(f"ⓘ Early stopping counter: {early_stop_counter}/{self.args.early_stop_patience}")
                # Check early stopping condition
                if early_stop_counter >= self.args.early_stop_patience:
                    print(
                        f"⏹ Early stopping triggered at Epoch {epoch + 1}, best validation loss: {best_vali_loss:.4f}")
                    break  # Exit training loop

            # Loss curve plotting
            if self.args.plot_loss_curve:
                mse_loss_records['train'].append(train_MSE_loss)
                mse_loss_records['valid'].append(vali_MSE_loss)
                mse_loss_records['test'].append(test_MSE_loss)
                save_loss_data(loss_dict=mse_loss_records, file_path='training_logs/Mse_loss_data.json')

        # Load best model after training completion
        self.model.load_state_dict(torch.load(best_model_path))
        print("✓ Best model weights loaded")
        print("Training complete!!!!!!!!")

        return self.model

    def vali(self, vali_data, vali_loader):
        """
        Validation phase

        Args:
            vali_data: Validation dataset
            vali_loader: Validation data loader

        Returns:
            tuple: (average MSE loss, average MAE loss)
        """
        vali_MSE_loss = []
        vali_MAE_loss = []

        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()  # Keep on CPU for loss calculation

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Forward pass
                outputs, _ = self.model(batch_x, batch_x_mark)
                outputs, batch_y = outputs.detach().cpu(), batch_y.detach().cpu()

                # Calculate losses
                mse_loss = mse_criterion(outputs, batch_y)
                mae_loss = mae_criterion(outputs, batch_y)

                vali_MSE_loss.append(mse_loss.item())
                vali_MAE_loss.append(mae_loss.item())

        # Calculate average losses
        vali_MSE_loss, vali_MAE_loss = np.average(vali_MSE_loss), np.average(vali_MAE_loss)
        self.model.train()  # Return to training mode
        return vali_MSE_loss, vali_MAE_loss

    def test(self, test_data, test_loader, epoch):
        """
        Test phase with optional visualization

        Args:
            test_data: Test dataset
            test_loader: Test data loader
            epoch: Current epoch number

        Returns:
            tuple: (average MSE loss, average MAE loss)
        """
        test_MSE_loss = []
        test_MAE_loss = []

        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        self.model.eval()
        with torch.no_grad():
            test_iter_count = 0  # Test iteration counter
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()  # Keep on CPU for loss calculation

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Forward pass
                outputs, _ = self.model(batch_x, batch_x_mark)
                outputs, batch_y = outputs.detach().cpu(), batch_y.detach().cpu()

                # Select samples for visualization (first sample in batch)
                batch_size = batch_y.shape[0]
                selected_indices = [0]  # First sample only
                plot_batch_y = batch_y[selected_indices]
                plot_outputs = outputs[selected_indices]
                test_iter_count += 1

                # Visualization condition
                if self.args.plot_result and epoch > 3 and plot_batch_y.shape[0] * (test_iter_count - 1) < 500:
                    plot_results(
                        batch_y=plot_batch_y,
                        preds=plot_outputs,
                        epoch_index=epoch + 1,
                        iter_count=test_iter_count,
                        task=f"{self.args.seq_len}_{self.args.pred_len}",
                        phase='test'
                    )

                # Calculate losses
                mse_loss = mse_criterion(outputs, batch_y)
                mae_loss = mae_criterion(outputs, batch_y)

                test_MSE_loss.append(mse_loss.item())
                test_MAE_loss.append(mae_loss.item())

        # Calculate average losses
        test_MSE_loss, test_MAE_loss = np.average(test_MSE_loss), np.average(test_MAE_loss)
        self.model.train()  # Return to training mode
        return test_MSE_loss, test_MAE_loss