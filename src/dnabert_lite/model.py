"""Small BERT-style encoder models for DNABERT-lite."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import MultiHeadSelfAttention


@dataclass
class DNABertLiteConfig:
    vocab_size: int
    max_position_embeddings: int = 512
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 256
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    pad_token_id: int = 0
    num_labels: int = 2


class DNABertLiteEmbeddings(nn.Module):
    """Token plus learned positional embeddings."""

    def __init__(self, config: DNABertLiteConfig) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(self.layer_norm(embeddings))


class DNABertLiteEncoderLayer(nn.Module):
    """One BERT-style encoder block."""

    def __init__(self, config: DNABertLiteConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout_prob,
        )
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.output_layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )
        attention_weights = None
        if return_attention:
            attention_output, attention_weights = attention_output

        hidden_states = self.attention_layer_norm(hidden_states + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.output_layer_norm(hidden_states + self.dropout(feed_forward_output))

        if return_attention:
            return hidden_states, attention_weights
        return hidden_states


class DNABertLiteEncoder(nn.Module):
    """Encoder-only DNABERT-lite backbone."""

    def __init__(self, config: DNABertLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = DNABertLiteEmbeddings(config)
        self.layers = nn.ModuleList(
            DNABertLiteEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attentions: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        hidden_states = self.embeddings(input_ids)
        attentions: list[torch.Tensor] = []

        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                return_attention=return_attentions,
            )
            if return_attentions:
                hidden_states, attention_weights = layer_output
                attentions.append(attention_weights)
            else:
                hidden_states = layer_output

        pooled_output = hidden_states[:, 0]
        output: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
        }
        if return_attentions:
            output["attentions"] = attentions
        return output


class DNABertLiteForMaskedLM(nn.Module):
    """DNABERT-lite encoder with a masked-language-modeling head."""

    def __init__(self, config: DNABertLiteConfig) -> None:
        super().__init__()
        self.encoder = DNABertLiteEncoder(config)
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.mlm_head(encoder_output["last_hidden_state"])
        output = {"logits": logits}

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output


class DNABertLiteForSequenceClassification(nn.Module):
    """DNABERT-lite encoder with a sequence classification head."""

    def __init__(self, config: DNABertLiteConfig) -> None:
        super().__init__()
        self.encoder = DNABertLiteEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(encoder_output["pooler_output"]))
        output = {"logits": logits, "embeddings": encoder_output["pooler_output"]}

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            output["loss"] = loss
        return output
