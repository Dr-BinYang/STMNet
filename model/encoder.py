import torch
import torch.nn as nn
import torch.nn.functional as F

from model.STFus import STFus
from model.attention import FullAttention, AttentionLayer
from model.series_decomp import series_decomp
from utils.plot_results import plot_attention_heatmap


class ConvLayer(nn.Module):
    """
    Convolutional layer with downsampling for feature extraction
    Uses circular padding to handle temporal boundaries
    """

    def __init__(self, c_in):
        """
        Initialize convolutional layer

        Args:
            c_in: Number of input channels
        """
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Forward pass of convolutional layer

        Args:
            x: Input tensor [batch_size, seq_len, channels]

        Returns:
            Downsampled features [batch_size, seq_len//2, channels]
        """
        x = self.downConv(x.permute(0, 2, 1))  # Convert to [batch_size, channels, seq_len]
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)  # Downsample by factor of 2
        x = x.transpose(1, 2)  # Convert back to [batch_size, seq_len, channels]
        return x


class My_Layernorm(nn.Module):
    """
    Custom layer normalization designed for seasonal components
    Removes mean bias after standard layer normalization
    """

    def __init__(self, channels):
        """
        Initialize custom layer normalization

        Args:
            channels: Number of feature channels
        """
        super(My_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Apply custom layer normalization

        Args:
            x: Input tensor [batch_size, seq_len, channels]

        Returns:
            Normalized tensor with mean bias removed
        """
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class EncoderLayer(nn.Module):
    """
    Single encoder layer with attention mechanism and feed-forward network
    Supports both STFus and standard attention
    """

    def __init__(self, args, activation="relu"):
        """
        Initialize encoder layer

        Args:
            args: Configuration arguments
            activation: Activation function type ('relu' or 'gelu')
        """
        super(EncoderLayer, self).__init__()
        self.args = args

        # Choose between STFus spatial-temporal fusion or standard attention
        if args.STFus == True:
            self.attention = STFus(args)
        else:
            self.attention = AttentionLayer(FullAttention(), d_model=args.d_model, n_heads=args.n_heads)

        # Feed-forward network components
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=args.d_ff, out_channels=args.d_model, kernel_size=1, bias=False)

        # Series decomposition modules for residual connections
        self.decomp1 = series_decomp(args.moving_avg)
        self.decomp2 = series_decomp(args.moving_avg)

        self.dropout = nn.Dropout(args.dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        """
        Forward pass of encoder layer

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Encoded features [batch_size, seq_len, d_model]
        """
        # Apply attention mechanism
        if self.args.STFus == True:
            new_x = self.attention(x)
        else:
            new_x, attn = self.attention(x, x, x)

        # Residual connection and series decomposition
        x = x + new_x
        x, _ = self.decomp1(x)

        # Feed-forward network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Second residual connection and decomposition
        enc_out, _ = self.decomp2(x + y)

        # Plot attention weights if enabled and using standard attention
        if self.args.plot_attn_weights == True and self.args.STFus == False:
            plot_attention_heatmap(attn)

        return enc_out


class Encoder(nn.Module):
    """
    Complete encoder module composed of multiple encoder layers
    Applies sequential processing with layer normalization
    """

    def __init__(self, args):
        """
        Initialize encoder module

        Args:
            args: Configuration arguments
        """
        super(Encoder, self).__init__()
        self.args = args

        # Stack of encoder layers
        self.encoderList = nn.ModuleList([EncoderLayer(args) for l in range(args.e_layers)])

        # Final layer normalization
        self.norm = My_Layernorm(args.d_model)

    def forward(self, x):
        """
        Forward pass through entire encoder

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Encoded representation [batch_size, seq_len, d_model]
        """
        # Process through each encoder layer
        for encoder_layer in self.encoderList:
            x = encoder_layer(x)

        # Apply final normalization
        x = self.norm(x)
        return x