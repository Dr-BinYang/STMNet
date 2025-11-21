import torch


class TriangularCausalMask():
    """
    Causal mask for autoregressive attention mechanisms
    Prevents positions from attending to subsequent positions in the sequence
    """

    def __init__(self, B, L, device="cpu"):
        """
        Initialize triangular causal mask

        Args:
            B: Batch size
            L: Sequence length
            device: Device to store the mask on
        """
        mask_shape = [B, 1, L, L]  # Shape: [batch_size, 1, seq_len, seq_len]
        with torch.no_grad():
            # Create upper triangular matrix with ones above the diagonal
            # triu(..., diagonal=1) sets the diagonal and below to 0, above to 1
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """Return the causal mask tensor"""
        return self._mask


class ProbMask():
    """
    Probabilistic mask for sparse attention mechanisms
    Used in efficient attention to limit the attention span
    """

    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Initialize probabilistic mask

        Args:
            B: Batch size
            H: Number of attention heads
            L: Sequence length
            index: Indices of selected queries for sparse attention
            scores: Attention scores tensor
            device: Device to store the mask on
        """
        # Create base triangular mask for single sequence
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # Expand mask to batch and head dimensions
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])

        # Select mask elements corresponding to the sampled query indices
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        # Reshape to match scores tensor shape
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """Return the probabilistic mask tensor"""
        return self._mask