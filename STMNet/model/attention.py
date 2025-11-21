import torch
import torch.nn as nn
import math
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models
    Adds positional information to input sequences using sine and cosine functions
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding

        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length to precompute
        """
        super().__init__()
        # Precompute positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Position-aware tensor of same shape as input
        """
        pe = self.pe[:x.size(1)]
        return x + pe


class FullAttention(nn.Module):
    """
    Standard multi-head attention mechanism
    Supports causal masking for autoregressive generation
    """

    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        """
        Initialize full attention module

        Args:
            mask_flag: Whether to apply causal masking
            factor: Unused parameter (for compatibility)
            scale: Scaling factor for attention scores
            attention_dropout: Dropout rate for attention weights
            output_attention: Whether to return attention weights
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Compute attention-weighted sum of values

        Args:
            queries: Query tensor [batch_size, seq_len, n_heads, d_k]
            keys: Key tensor [batch_size, seq_len, n_heads, d_k]
            values: Value tensor [batch_size, seq_len, n_heads, d_v]
            attn_mask: Attention mask (optional)

        Returns:
            tuple: (attention output, attention weights)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Compute attention scores using Einstein summation
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply causal masking if required
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Compute attention weights and apply dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # Compute weighted sum of values
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """
    ProbSparse attention mechanism for efficient long sequence processing
    Reduces computational complexity by sampling important queries
    """

    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        Initialize probabilistic sparse attention

        Args:
            mask_flag: Whether to apply causal masking
            factor: Sampling factor for sparse attention
            scale: Scaling factor for attention scores
            attention_dropout: Dropout rate for attention weights
            output_attention: Whether to return attention weights
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Sample top-k queries based on sparsity measurement

        Args:
            Q: Query tensor [batch_size, n_heads, seq_len_q, d_k]
            K: Key tensor [batch_size, n_heads, seq_len_k, d_k]
            sample_k: Number of keys to sample
            n_top: Number of top queries to select

        Returns:
            tuple: (sampled attention scores, indices of top queries)
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample random keys for each query
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Compute sparsity measurement and select top-k queries
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Compute attention scores for top queries only
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        Initialize context with mean or cumulative sum of values

        Args:
            V: Value tensor [batch_size, n_heads, seq_len_v, d_v]
            L_Q: Length of query sequence

        Returns:
            Initial context tensor
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            # For masked attention, use cumulative sum
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        Update context with attention from selected queries

        Args:
            context_in: Initial context tensor
            V: Value tensor
            scores: Attention scores for top queries
            index: Indices of top queries
            L_Q: Query sequence length
            attn_mask: Attention mask

        Returns:
            Updated context tensor
        """
        B, H, L_V, D = V.shape

        # Apply masking if required
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Compute attention weights
        attn = torch.softmax(scores, dim=-1)

        # Update context with attention-weighted values
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            # Create full attention matrix for visualization
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass of probabilistic sparse attention

        Args:
            queries: Query tensor [batch_size, seq_len, n_heads, d_k]
            keys: Key tensor [batch_size, seq_len, n_heads, d_k]
            values: Value tensor [batch_size, seq_len, n_heads, d_v]
            attn_mask: Attention mask

        Returns:
            tuple: (attention output, attention weights)
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose for head-first format
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Determine number of queries and keys to sample
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # Sample top queries and compute their attention scores
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Apply scaling
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Initialize and update context
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class ReformerLayer(nn.Module):
    """
    Reformer layer using Locality-Sensitive Hashing (LSH) attention
    Efficient attention mechanism for long sequences
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        """
        Initialize Reformer layer

        Args:
            attention: Attention type (for compatibility, not used)
            d_model: Model dimension
            n_heads: Number of attention heads
            d_keys: Key dimension (unused)
            d_values: Value dimension (unused)
            causal: Whether to use causal masking
            bucket_size: Size of buckets for LSH attention
            n_hashes: Number of hashes for LSH attention
        """
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        """
        Pad sequence length to be divisible by bucket_size * 2

        Args:
            queries: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Padded tensor with appropriate length
        """
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass of Reformer layer

        Args:
            queries: Query tensor
            keys: Key tensor
            values: Value tensor
            attn_mask: Attention mask

        Returns:
            tuple: (attention output, None) - Reformer doesn't return attention weights
        """
        B, N, C = queries.shape
        # Apply LSH attention with length adjustment
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class AttentionLayer(nn.Module):
    """
    Complete attention layer with projections and positional encoding
    Wraps different attention mechanisms with linear projections
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        """
        Initialize attention layer

        Args:
            attention: Attention mechanism (FullAttention, ProbAttention, etc.)
            d_model: Model dimension
            n_heads: Number of attention heads
            d_keys: Dimension of keys (default: d_model // n_heads)
            d_values: Dimension of values (default: d_model // n_heads)
        """
        super(AttentionLayer, self).__init__()
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        # Linear projections for queries, keys, and values
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Forward pass of attention layer

        Args:
            queries: Query tensor [batch_size, seq_len_q, d_model]
            keys: Key tensor [batch_size, seq_len_k, d_model]
            values: Value tensor [batch_size, seq_len_v, d_model]
            attn_mask: Attention mask

        Returns:
            tuple: (attention output, attention weights)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project inputs to key, query, value space
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # Add positional encoding and reshape for multi-head attention
        queries = self.pos_encoder(queries).view(B, L, H, -1)
        keys = self.pos_encoder(keys).view(B, S, H, -1)
        values = self.pos_encoder(values).view(B, S, H, -1)

        # Apply attention mechanism
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        # Reshape and project back to model dimension
        out = out.view(B, L, -1)
        return self.out_projection(out), attn