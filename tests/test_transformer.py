import torch

from transformer_pytorch import (
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
    TransformerConfig,
    build_causal_mask,
)


def small_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=50,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        max_len=16,
    )


def test_transformer_forward_shape() -> None:
    model = Transformer(small_config())
    src = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    tgt = torch.tensor([[1, 7, 8], [1, 9, 0]])

    logits = model(src, tgt)

    assert logits.shape == (2, 3, 50)


def test_positional_encoding_changes_equal_tokens_by_position() -> None:
    position = PositionalEncoding(d_model=8, max_len=8, dropout=0.0)
    x = torch.zeros(1, 4, 8)

    encoded = position(x)

    assert not torch.allclose(encoded[:, 0], encoded[:, 1])


def test_causal_mask_blocks_future_attention() -> None:
    attention = MultiHeadAttention(d_model=8, num_heads=2, dropout=0.0)
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    v = torch.randn(1, 3, 8)
    mask = build_causal_mask(seq_len=3, device=q.device)

    _, weights = attention(q, k, v, mask=mask, return_weights=True)

    assert torch.all(weights[:, :, 0, 1:] == 0)
    assert torch.all(weights[:, :, 1, 2:] == 0)
