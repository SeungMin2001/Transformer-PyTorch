import torch

from transformer_pytorch import Transformer, TransformerConfig


def main() -> None:
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        max_len=32,
    )
    model = Transformer(config)

    src_tokens = torch.tensor([[5, 7, 9, 0], [4, 3, 2, 1]])
    tgt_tokens = torch.tensor([[1, 8, 6], [1, 2, 3]])
    logits = model(src_tokens, tgt_tokens)

    print("logits shape:", tuple(logits.shape))
    print("expected shape: (batch=2, target_len=3, vocab_size=100)")


if __name__ == "__main__":
    main()
