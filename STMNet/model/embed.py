import torch
import torch.nn as nn
import math


def compared_version(ver1, ver2):
    """
    Compare two version strings

    Args:
        ver1: First version string
        ver2: Second version string

    Returns:
        int: -1 if ver1 < ver2, 1 if ver1 > ver2, True if equal
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")

    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1

    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True


class FixedEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding
    Uses precomputed sinusoidal patterns that don't require gradient updates
    """

    def __init__(self, c_in, d_model):
        """
        Initialize fixed embedding layer

        Args:
            c_in: Input dimension (vocabulary size)
            d_model: Output embedding dimension
        """
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """Apply fixed sinusoidal embedding"""
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    """
    Time feature embedding using linear projection
    Embeds temporal features like hour, day, etc.
    """

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        Initialize time feature embedding

        Args:
            d_model: Output embedding dimension
            embed_type: Embedding type (not used in current implementation)
            freq: Frequency of time features ('h', 't', 's', 'm', 'a', 'w', 'd', 'b')
        """
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """Embed time features using linear projection"""
        return self.embed(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution
    Projects input tokens to embedding space with convolutional processing
    """

    def __init__(self, c_in, d_model):
        """
        Initialize token embedding layer

        Args:
            c_in: Number of input channels/features
            d_model: Output embedding dimension
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Initialize weights using Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """Apply convolutional token embedding"""
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information
    Uses fixed sinusoidal patterns for positional encoding
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional embedding

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length to precompute
        """
        super(PositionalEmbedding, self).__init__()
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Return positional encodings for input sequence length"""
        return self.pe[:, :x.size(1)]


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time-related features
    Embeds minute, hour, weekday, day, and month components separately
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        Initialize temporal embedding

        Args:
            d_model: Embedding dimension
            embed_type: Embedding type (not used in current implementation)
            freq: Frequency indicator
        """
        super(TemporalEmbedding, self).__init__()

        minute_size = 6
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # Use learnable embeddings (original code had conditional FixedEmbedding)
        Embed = nn.Embedding

        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        Embed temporal features by summing individual component embeddings

        Args:
            x: Input tensor with temporal features [batch_size, seq_len, 5]
                where last dimension: [month, day, weekday, hour, minute]
        """
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4])
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without positional encoding
    Combines value embedding and temporal embedding
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Initialize data embedding module

        Args:
            c_in: Number of input features
            d_model: Embedding dimension
            embed_type: Embedding type
            freq: Frequency indicator for temporal features
            dropout: Dropout rate
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Apply data embedding to input and temporal markers

        Args:
            x: Input data [batch_size, seq_len, c_in]
            x_mark: Temporal markers [batch_size, seq_len, temporal_features]
        """
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    """
    Data embedding without positional encoding and temporal information
    Uses only value embedding for trend components
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Initialize simplified data embedding (value only)

        Args:
            c_in: Number of input features
            d_model: Embedding dimension
            embed_type: Embedding type (unused)
            freq: Frequency indicator (unused)
            dropout: Dropout rate
        """
        super(DataEmbedding_wo_pos_temp, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Apply value embedding only (for trend components)"""
        x = self.value_embedding(x)
        return self.dropout(x)