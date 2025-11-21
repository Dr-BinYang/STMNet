import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    Uses average pooling with padding to maintain sequence length
    """

    def __init__(self, kernel_size, stride):
        """
        Initialize moving average module

        Args:
            kernel_size: Size of the moving average window
            stride: Stride for the average pooling operation
        """
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        Apply moving average to input time series

        Args:
            x: Input tensor [batch_size, seq_len, features]

        Returns:
            Smoothed time series [batch_size, seq_len, features]
        """
        # Pad both ends of the time series to maintain length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # Apply average pooling and restore dimensions
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    Decomposes time series into seasonal and trend components
    """

    def __init__(self, kernel_size):
        """
        Initialize series decomposition module

        Args:
            kernel_size: Window size for moving average decomposition
        """
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        Decompose input time series into seasonal and trend components

        Args:
            x: Input time series [batch_size, seq_len, features]

        Returns:
            tuple: (seasonal_component, trend_component)
                   seasonal: High-frequency fluctuations [batch_size, seq_len, features]
                   trend: Low-frequency trend [batch_size, seq_len, features]
        """
        # Calculate moving average as trend component
        moving_mean = self.moving_avg(x)

        # Calculate residual as seasonal component
        res = x - moving_mean

        return res, moving_mean  # Seasonal, Trend