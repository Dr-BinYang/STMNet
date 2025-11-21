import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # Optional: suppress all user warnings
import math


class SpatialModule(nn.Module):
    """
    Spatial Module for modeling spatial dependencies using graph attention networks
    Dynamically generates graph edges and applies GATv2 convolution
    """

    def __init__(self, args, num_nodes=32, heads=1, topk=20):
        """
        Initialize Spatial Module

        Args:
            args: Configuration arguments
            num_nodes: Number of nodes in the graph
            heads: Number of attention heads
            topk: Top K edges to keep for sparsity
        """
        super().__init__()
        d_model = args.d_model
        self.num_nodes = num_nodes
        self.node_dim = d_model // num_nodes
        self.heads = heads
        self.topk = topk
        assert d_model % num_nodes == 0, "d_model must be divisible by num_nodes"

        # Dynamic edge generator
        self.edge_gen = nn.Sequential(
            nn.Linear(2 * self.node_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Learnable base edge weight
        self.base_edge_weight = nn.Parameter(torch.ones(1))

        # Multi-head graph attention
        self.gat = GATv2Conv(
            in_channels=self.node_dim,
            out_channels=self.node_dim,
            heads=heads,
            concat=True,
            edge_dim=1,
            dropout=0.1
        )
        self.proj = nn.Linear(heads * self.node_dim, self.node_dim)
        self.norm = nn.LayerNorm(d_model)

    def _get_batch_edges(self, x_nodes):
        """
        Dynamically generate batch edge indices and features with TopK filtering

        Args:
            x_nodes: Node features [B, N, D]

        Returns:
            tuple: (edge_index, edge_attr) - edge indices and attributes
        """
        B, N, D = x_nodes.shape  # B = batch * time (samples per graph)

        # Generate all possible node pairs (excluding self-loops)
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        single_edge_index = torch.tensor([rows, cols], device=x_nodes.device)  # [2, E]
        E = single_edge_index.size(1)

        # Get source and destination node features
        src = x_nodes[:, single_edge_index[0]]  # [B, E, D]
        dst = x_nodes[:, single_edge_index[1]]  # [B, E, D]
        edge_attr = self.edge_gen(torch.cat([src, dst], dim=-1))  # [B, E, 1]

        # TopK filtering: select top K edges for each graph
        if self.topk < E:
            topk_mask = torch.zeros(B, E, dtype=torch.bool, device=x_nodes.device)
            for b in range(B):
                topk_indices = torch.topk(edge_attr[b, :, 0], self.topk).indices
                topk_mask[b, topk_indices] = True

            edge_attr = edge_attr[topk_mask].view(-1, 1)  # [B*K, 1]
            edge_index_list = []
            for b in range(B):
                topk_idx = topk_mask[b].nonzero(as_tuple=False).squeeze(1)
                offset = b * N  # Offset to ensure unique node numbering across graphs
                edge_index_b = single_edge_index[:, topk_idx] + offset
                edge_index_list.append(edge_index_b)
            edge_index = torch.cat(edge_index_list, dim=1)  # [2, B*K]
        else:
            # No filtering: replicate edge_index for all graphs with offsets
            edge_index = single_edge_index.repeat(1, B)
            offsets = torch.arange(B, device=x_nodes.device).view(1, -1).repeat(E, 1).T * N  # [B, E]
            edge_index = edge_index.view(2, 1, E).repeat(1, B, 1) + offsets.unsqueeze(0)
            edge_index = edge_index.view(2, -1)
            edge_attr = edge_attr.view(-1, 1)

        return edge_index, edge_attr

    def forward(self, x):
        """
        Forward pass of Spatial Module

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        B, L, _ = x.shape
        total_samples = B * L

        # Reshape each sample into graph nodes [total_samples, num_nodes, node_dim]
        x_nodes = x.reshape(total_samples, self.num_nodes, self.node_dim)
        edge_index, edge_attr = self._get_batch_edges(x_nodes)

        # Flatten and apply graph attention
        x_flat = x_nodes.view(-1, self.node_dim)
        out = self.gat(x_flat, edge_index, edge_attr)

        # Project and reshape back
        out = self.proj(out)
        out = out.view(total_samples, self.num_nodes, self.node_dim)

        # Restore original shape [B, L, d_model]
        return self.norm(out.view(B, L, -1))


class TemporalModule(nn.Module):
    """
    Temporal Module for modeling temporal dependencies
    Supports GRU and Attention-based approaches
    """

    def __init__(self, args):
        """
        Initialize Temporal Module

        Args:
            args: Configuration arguments
        """
        super().__init__()
        d_model = args.d_model
        self.mode = args.STFusTempor

        # Shared components
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Mode-specific branches
        if self.mode == 'GRU':
            self.temporal_layer = nn.GRU(
                input_size=2 * d_model,
                hidden_size=d_model,
                num_layers=2,
                batch_first=True,
                bidirectional=False
            )
        elif self.mode == 'Attention':
            self.num_heads = 8
            self.proj_input = nn.Linear(2 * d_model, d_model)
            self.temporal_layer = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=self.num_heads,
                batch_first=True
            )
            self.pos_encoder = PositionalEncoding(d_model)
        else:
            raise ValueError(f"Unsupported temporal mode: {self.mode}")

    def forward(self, x, spatial_out):
        """
        Forward pass of Temporal Module

        Args:
            x: Original input [batch_size, seq_len, d_model]
            spatial_out: Spatial module output [batch_size, seq_len, d_model]

        Returns:
            Temporal features [batch_size, seq_len, d_model]
        """
        # Concatenate input and spatial output
        combined = torch.cat([x, spatial_out], dim=-1)

        if self.mode == 'GRU':
            output, _ = self.temporal_layer(combined)
        else:
            # Project to appropriate dimension
            attn_input = self.proj_input(combined)

            # Add positional encoding
            attn_input = self.pos_encoder(attn_input)

            # Apply multi-head attention
            output, _ = self.temporal_layer(attn_input, attn_input, attn_input)

        return self.norm(self.proj(output))


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for sequence models using sine and cosine functions
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize Positional Encoding

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Position-aware tensor [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        pe = self.pe[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
        return x + pe.to(x.device)


class GlobalModule(nn.Module):
    """Global feature module using feed-forward network"""

    def __init__(self, args):
        """
        Initialize Global Module

        Args:
            args: Configuration arguments
        """
        super().__init__()
        d_model = args.d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 1 * d_model),
            nn.GELU(),
            nn.Linear(1 * d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        """Apply feed-forward network to input"""
        return self.ffn(x)


class ChannelAttention(nn.Module):
    """Pixel-wise channel attention with position-independent weights"""

    def __init__(self, channels=3, reduction=3):
        """
        Initialize Channel Attention module

        Args:
            channels: Number of input channels
            reduction: Channel reduction factor
        """
        super().__init__()
        assert channels >= reduction, f"Number of channels({channels}) must be ≥ reduction factor({reduction})"

        # Define layers
        self.conv_reduce = nn.Conv2d(
            in_channels=channels,
            out_channels=channels // reduction,
            kernel_size=1  # Preserve spatial dimensions
        )
        self.act = nn.ReLU()
        self.conv_expand = nn.Conv2d(
            in_channels=channels // reduction,
            out_channels=channels,
            kernel_size=1
        )
        self.softmax = nn.Softmax(dim=1)  # Channel dimension

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization"""
        nn.init.kaiming_normal_(self.conv_reduce.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_reduce.bias, 0)
        nn.init.xavier_uniform_(self.conv_expand.weight)
        nn.init.constant_(self.conv_expand.bias, 0)

    def forward(self, x):
        """
        Forward pass of Channel Attention

        Args:
            x: Input tensor [B, C, L, D]

        Returns:
            Weighted output [B, L, D]
        """
        # Channel reduction
        reduced = self.conv_reduce(x)
        reduced = self.act(reduced)

        # Channel restoration
        weights = self.conv_expand(reduced)
        weights = self.softmax(weights)  # [B, C, L, D]

        # Pixel-wise weighted fusion
        return torch.sum(x * weights, dim=1)  # Sum along channel dimension


class STFus(nn.Module):
    """Final Spatio-Temporal Fusion model integrating spatial, temporal and global modules"""

    def __init__(self, args):
        """
        Initialize STFus model

        Args:
            args: Configuration arguments
        """
        super().__init__()
        self.args = args
        self.spatial = SpatialModule(args)
        self.temporal = TemporalModule(args)
        self.global_net = GlobalModule(args)
        self.channel_attn = ChannelAttention()

    def forward(self, x):
        """
        Forward pass of STFus model

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Fused spatio-temporal features [batch_size, seq_len, d_model]
        """
        identity = x  # Residual connection

        # Three-path processing
        spatial_out = self.spatial(x)
        temporal_out = self.temporal(x, spatial_out)
        global_out = self.global_net(x)

        # Channel fusion
        fused = torch.stack([spatial_out, temporal_out, global_out], dim=1)
        output = self.channel_attn(fused)

        return output


# Test code
if __name__ == '__main__':
    class Config:
        """Mock configuration class for testing"""
        d_model = 1024
        num_nodes = 32  # 1024/32=32
        gat_heads = 4


    # Test spatial module
    def test_spatial():
        """Test function for SpatialModule"""
        module = SpatialModule(1024)
        x = torch.randn(2, 100, 1024)
        try:
            out = module(x)
            assert out.shape == x.shape
            print("SpatialModule test passed!")
        except Exception as e:
            print(f"SpatialModule test failed: {e}")


    test_spatial()

    # Full model test
    model = STFus(Config())
    x = torch.randn(32, 192, 1024)
    try:
        output = model(x)
        print(f"\nFull model test passed! Input-output dimensions match: {x.shape} → {output.shape}")
    except Exception as e:
        print(f"Model error: {e}")