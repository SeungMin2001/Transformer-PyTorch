from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters used by the original paper and the README examples."""

    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1
    max_len: int = 512
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


def build_padding_mask(tokens: Tensor, pad_token_id: int = 0) -> Tensor:
    """Return a mask with shape [batch, 1, 1, seq_len], where True means usable."""

    return tokens.ne(pad_token_id).unsqueeze(1).unsqueeze(2)


def build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Return a lower-triangular decoder mask with shape [1, 1, seq_len, seq_len]."""

    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).view(
        1, 1, seq_len, seq_len
    )


class TokenEmbedding(nn.Module):
    """Token embedding multiplied by sqrt(d_model), as described in the paper."""

    def __init__(self, vocab_size: int, d_model: int, pad_token_id: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    The README notes that token embedding alone has no order information.
    This module adds fixed sin/cos vectors so the model can distinguish token positions.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention split into multiple heads."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        output = self.out_proj(self._combine_heads(self.dropout(context)))

        if return_weights:
            return output, weights
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network: d_model -> d_ff -> d_model."""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """One encoder block: self-attention, residual Add & Norm, then feed-forward."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        attn = self.self_attention(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn))
        ff = self.feed_forward(src)
        return self.norm2(src + self.dropout(ff))


class DecoderLayer(nn.Module):
    """One decoder block with masked self-attention and encoder-decoder attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.cross_attention = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        self_attn = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn))

        cross_attn = self.cross_attention(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn))

        ff = self.feed_forward(tgt)
        return self.norm3(tgt + self.dropout(ff))


class Transformer(nn.Module):
    """Encoder-decoder Transformer for sequence-to-sequence learning."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(
            config.vocab_size, config.d_model, config.pad_token_id
        )
        self.position = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    def encode(self, src_tokens: Tensor, src_mask: Tensor | None = None) -> Tensor:
        x = self.position(self.token_embedding(src_tokens))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.position(self.token_embedding(tgt_tokens))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        if src_mask is None:
            src_mask = build_padding_mask(src_tokens, self.config.pad_token_id)
        if tgt_mask is None:
            tgt_pad_mask = build_padding_mask(tgt_tokens, self.config.pad_token_id)
            tgt_mask = tgt_pad_mask & build_causal_mask(tgt_tokens.size(1), tgt_tokens.device)

        memory = self.encode(src_tokens, src_mask)
        decoder_state = self.decode(tgt_tokens, memory, tgt_mask, src_mask)
        return self.output_projection(decoder_state)
