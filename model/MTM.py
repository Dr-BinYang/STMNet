import torch
import torch.nn as nn
import torch.nn.functional as Ffunc


class MTM(nn.Module):
    """
    Multi-scale Temporal Memory (MTM) module
    Decomposes time series into multi-scale trends and periodic components
    and fuses them hierarchically
    """

    def __init__(self, feature_dim, d_model, M=4):
        """
        Initialize MTM module

        Args:
            feature_dim: Input feature dimension
            d_model: Output model dimension
            M: Number of decomposition scales
        """
        super(MTM, self).__init__()
        self.d_model = d_model
        self.M = M
        self.feature_dim = feature_dim

        # Learnable linear projections for multi-scale trend and periodic components (preserving feature dimension)
        self.trend_weights = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(M - 1)
        ])
        self.period_weights = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(M - 1)
        ])

        # Project each scale's trend and fused components to d_model dimension
        self.linear_T = nn.ModuleList([
            nn.Linear(feature_dim, d_model) for _ in range(M)
        ])
        self.linear_X = nn.ModuleList([
            nn.Linear(feature_dim, d_model) for _ in range(M)
        ])

        # Convolutional fusion along the scale dimension M, output channel=1 (fused to single scale)
        self.conv_T = nn.Conv1d(M, 1, kernel_size=1)
        self.conv_X = nn.Conv1d(M, 1, kernel_size=1)

    def decompose(self, x, scale):
        """
        Multi-scale time series decomposition

        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]
            scale: Current scale (determines avgpool kernel and stride)

        Returns:
            tuple: (trend, period)
                trend: Smoothed trend component [batch_size, L', feature_dim]
                period: Periodic residual component [batch_size, L'', feature_dim]
        """
        kernel = stride = 2 ** scale
        x_t = x.transpose(1, 2)  # Convert to [batch_size, feature_dim, seq_len]

        # Apply average pooling for trend extraction
        trend = Ffunc.avg_pool1d(x_t, kernel_size=kernel, stride=stride, ceil_mode=True)

        # Interpolate trend back to original sequence length
        trend_interpo = Ffunc.interpolate(trend, size=x.shape[1], mode='linear', align_corners=False)
        trend_interpo = trend_interpo.transpose(1, 2)
        trend = trend.transpose(1, 2)

        # Calculate periodic component as residual
        period = x - trend_interpo
        period = period.transpose(1, 2)
        period = Ffunc.avg_pool1d(period, kernel_size=kernel, stride=stride, ceil_mode=True).transpose(1, 2)

        return trend, period

    def forward(self, x):
        """
        Forward pass of MTM module

        Args:
            x: Input tensor [batch_size, seq_len, feature_dim]

        Returns:
            tuple: (X_hat, T_hat) - fused features and trend components
        """
        B, L, feature_dim = x.shape
        trends, periods = [], []

        # Step 1: Multi-scale decomposition
        for m in range(self.M):
            T_m, P_m = self.decompose(x, m)
            trends.append(T_m)
            periods.append(P_m)

        # Step 2: Coarse-to-fine trend fusion
        for m in reversed(range(self.M - 1)):
            delta = Ffunc.gelu(self.trend_weights[m](trends[m + 1]))
            delta = Ffunc.interpolate(delta.transpose(1, 2), size=trends[m].shape[1], mode='linear',
                                      align_corners=False).transpose(1, 2)
            trends[m] = trends[m] + delta

        # Step 3: Fine-to-coarse periodic fusion
        for m in range(1, self.M):
            delta = Ffunc.gelu(self.period_weights[m - 1](periods[m - 1]))
            delta = Ffunc.avg_pool1d(delta.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).transpose(1, 2)
            delta = Ffunc.interpolate(delta.transpose(1, 2), size=periods[m].shape[1], mode='linear',
                                      align_corners=False).transpose(1, 2)
            periods[m] = periods[m] + delta

        # Step 4: Per-scale fusion: trend + period → [batch_size, seq_len, d_model]
        x_temp, T_temp = [], []
        for m in range(self.M):
            # Align dimensions if needed
            if trends[m].shape[1] != periods[m].shape[1]:
                periods[m] = Ffunc.interpolate(periods[m].transpose(1, 2), size=trends[m].shape[1], mode='linear',
                                               align_corners=False).transpose(1, 2)

            # Project trend and fused components
            T_m_d = self.linear_T[m](trends[m])
            S_m_d = self.linear_X[m](trends[m] + periods[m])

            # Interpolate to original sequence length
            T_m_d = Ffunc.interpolate(T_m_d.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            S_m_d = Ffunc.interpolate(S_m_d.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)

            x_temp.append(S_m_d)
            T_temp.append(T_m_d)

        # Step 5: Stack to [batch_size, M, seq_len, d_model]
        x_temp = torch.stack(x_temp, dim=1)
        T_temp = torch.stack(T_temp, dim=1)

        # Step 6: Convolutional fusion along scale dimension M
        # Rearrange to [batch_size, seq_len, d_model, M] → [batch_size*seq_len*d_model, M, 1]
        x_temp_perm = x_temp.permute(0, 2, 3, 1).reshape(-1, x_temp.size(1), 1)
        T_temp_perm = T_temp.permute(0, 2, 3, 1).reshape(-1, T_temp.size(1), 1)

        # Convolutional fusion → [batch_size*seq_len*d_model] → reshape to [batch_size, seq_len, d_model]
        X_hat = self.conv_X(x_temp_perm).squeeze(-1).squeeze(-1).reshape(B, L, self.d_model)
        T_hat = self.conv_T(T_temp_perm).squeeze(-1).squeeze(-1).reshape(B, L, self.d_model)

        return X_hat, T_hat


if __name__ == '__main__':
    # Test configuration
    class Config:
        d_model = 1024
        L = 96  # Sequence lengths: 96, 192, 336, 720
        B = 32
        feature_dim = 21
        M = 6


    # Test the MTM module
    x = torch.randn(Config.B, Config.L, Config.feature_dim)
    model = MTM(feature_dim=Config.feature_dim, d_model=Config.d_model, M=Config.M)
    X_hat, T_hat = model(x)
    print("X_hat shape:", X_hat.shape)
    print("T_hat shape:", T_hat.shape)