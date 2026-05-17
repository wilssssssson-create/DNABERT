"""Hand-written self-attention modules for DNABERT-lite."""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Args:
        query: Tensor with shape ``(batch, heads, seq_len, head_dim)``.
        key: Tensor with shape ``(batch, heads, seq_len, head_dim)``.
        value: Tensor with shape ``(batch, heads, seq_len, head_dim)``.
        attention_mask: Optional mask broadcastable to
            ``(batch, heads, seq_len, seq_len)``. Values of 0 are masked.

    Returns:
        A pair ``(context, attention_weights)``.
    """

    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

    if attention_mask is not None:
        mask = attention_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, value)
    return context, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """Minimal multi-head self-attention implemented from Q/K/V projections."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]

        context, attention_weights = scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
        )
        context = self._merge_heads(context)
        output = self.output(self.dropout(context))

        if return_attention:
            return output, attention_weights
        return output
