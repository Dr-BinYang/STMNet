import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import FullAttention, AttentionLayer
from model.series_decomp import series_decomp
from model.STFus import STFus


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
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
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


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and feed-forward network
    Supports both STFus and standard attention mechanisms
    """

    def __init__(self, args, activation="relu"):
        """
        Initialize decoder layer

        Args:
            args: Configuration arguments
            activation: Activation function type ('relu' or 'gelu')
        """
        super(DecoderLayer, self).__init__()
        self.args = args

        # Self-attention mechanism (STFus or standard attention)
        if args.STFus == True:
            self.self_attention = STFus(args)
        else:
            self.self_attention = AttentionLayer(FullAttention(), d_model=args.d_model, n_heads=args.n_heads)

        # Cross-attention with encoder outputs
        self.cross_attention = AttentionLayer(FullAttention(), d_model=args.d_model, n_heads=args.n_heads)

        # Feed-forward network components
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=args.d_ff, out_channels=args.d_model, kernel_size=1, bias=False)

        # Series decomposition modules for residual connections
        self.decomp1 = series_decomp(args.moving_avg)
        self.decomp2 = series_decomp(args.moving_avg)
        self.decomp3 = series_decomp(args.moving_avg)

        # Projection layer for trend components
        self.projection = nn.Conv1d(in_channels=args.d_model, out_channels=args.c_out, kernel_size=3, stride=1,
                                    padding=1,
                                    padding_mode='circular', bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        """
        Forward pass of decoder layer

        Args:
            x: Decoder input [batch_size, seq_len, d_model]
            cross: Encoder output for cross-attention [batch_size, seq_len, d_model]

        Returns:
            tuple: (processed features, residual trend component)
        """
        # Self-attention with residual connection
        if self.args.STFus == True:
            x = x + self.self_attention(x)
        else:
            x = x + self.self_attention(x, x, x, attn_mask=None)[0]

        # First decomposition
        x, trend1 = self.decomp1(x)

        # Cross-attention with encoder outputs
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=None
        )[0])

        # Second decomposition
        x, trend2 = self.decomp2(x)

        # Feed-forward network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Third decomposition
        x, trend3 = self.decomp3(x + y)

        # Combine trend components and project to output dimension
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class Decoder(nn.Module):
    """
    Complete decoder module composed of multiple decoder layers
    Generates predictions by combining seasonal and trend components hierarchically
    """

    def __init__(self, args):
        """
        Initialize decoder module

        Args:
            args: Configuration arguments
        """
        super(Decoder, self).__init__()
        self.args = args

        # Stack of decoder layers
        self.decoderList = nn.ModuleList([DecoderLayer(args) for l in range(args.d_layers)])

        # Final normalization and projection
        self.norm = My_Layernorm(args.d_model)
        self.projection = nn.Linear(args.d_model, args.c_out, bias=True)

    def forward(self, x, enc_out, trend):
        """
        Forward pass through entire decoder

        Args:
            x: Decoder input [batch_size, pred_len, d_model]
            enc_out: Encoder output [batch_size, seq_len, d_model]
            trend: Initial trend prediction [batch_size, pred_len, c_out]

        Returns:
            tuple: (seasonal_component, final_trend, hierarchical_predictions)
        """
        # Initialize hierarchy prediction list with initial trend
        hierarchy_prediction = [trend.detach().clone()]

        # Process through each decoder layer
        for decoder_layer in self.decoderList:
            x, residual_trend = decoder_layer(x, enc_out)
            trend = trend + residual_trend
            hierarchy_prediction.append(trend.detach().clone())  # Store trend at each layer

        # Final normalization and projection for seasonal component
        x = self.norm(x)
        x = self.projection(x)

        # Stack hierarchical predictions
        hierarchy_stack = torch.stack(hierarchy_prediction, dim=0)

        return x, trend, hierarchy_prediction