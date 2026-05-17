import torch

from dnabert_lite import (
    DNABertLiteConfig,
    DNABertLiteEncoder,
    DNABertLiteForMaskedLM,
    DNABertLiteForSequenceClassification,
)


def tiny_config() -> DNABertLiteConfig:
    return DNABertLiteConfig(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        pad_token_id=0,
        num_labels=2,
    )


def test_encoder_forward_shapes():
    config = tiny_config()
    model = DNABertLiteEncoder(config)
    input_ids = torch.tensor([[2, 4, 5, 0], [2, 6, 7, 8]])

    output = model(input_ids, return_attentions=True)

    assert output["last_hidden_state"].shape == (2, 4, config.hidden_size)
    assert output["pooler_output"].shape == (2, config.hidden_size)
    assert len(output["attentions"]) == config.num_hidden_layers
    assert output["attentions"][0].shape == (2, config.num_attention_heads, 4, 4)


def test_masked_lm_forward_with_loss():
    config = tiny_config()
    model = DNABertLiteForMaskedLM(config)
    input_ids = torch.tensor([[2, 4, 5, 0], [2, 6, 7, 8]])
    labels = torch.tensor([[-100, 4, -100, -100], [-100, -100, 7, -100]])

    output = model(input_ids, labels=labels)

    assert output["logits"].shape == (2, 4, config.vocab_size)
    assert output["loss"].ndim == 0


def test_sequence_classification_forward_with_loss():
    config = tiny_config()
    model = DNABertLiteForSequenceClassification(config)
    input_ids = torch.tensor([[2, 4, 5, 0], [2, 6, 7, 8]])
    labels = torch.tensor([0, 1])

    output = model(input_ids, labels=labels)

    assert output["logits"].shape == (2, config.num_labels)
    assert output["embeddings"].shape == (2, config.hidden_size)
    assert output["loss"].ndim == 0
