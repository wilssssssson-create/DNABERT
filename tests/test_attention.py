import torch

from dnabert_lite.attention import MultiHeadSelfAttention, scaled_dot_product_attention


def test_scaled_dot_product_attention_shapes():
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 5, 8)
    value = torch.randn(2, 4, 5, 8)

    context, weights = scaled_dot_product_attention(query, key, value)

    assert context.shape == (2, 4, 5, 8)
    assert weights.shape == (2, 4, 5, 5)


def test_multi_head_self_attention_masks_padding_tokens():
    attention = MultiHeadSelfAttention(hidden_size=16, num_heads=4, dropout=0.0)
    hidden_states = torch.randn(2, 5, 16)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

    output, weights = attention(hidden_states, attention_mask, return_attention=True)

    assert output.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)
    assert torch.allclose(weights[0, :, :, 3:], torch.zeros_like(weights[0, :, :, 3:]))
    assert torch.allclose(weights[1, :, :, 2:], torch.zeros_like(weights[1, :, :, 2:]))
