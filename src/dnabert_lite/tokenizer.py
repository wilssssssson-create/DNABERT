"""DNA k-mer tokenizer utilities."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Iterable, Sequence


class KmerTokenizer:
    """A compact BERT-style tokenizer for DNA k-mers."""

    pad_token = "[PAD]"
    mask_token = "[MASK]"
    cls_token = "[CLS]"
    unk_token = "[UNK]"

    def __init__(self, k: int = 6, vocab: dict[str, int] | None = None) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        if vocab is None:
            tokens = [self.pad_token, self.mask_token, self.cls_token, self.unk_token]
            tokens.extend("".join(kmer) for kmer in itertools.product("ACGT", repeat=k))
            vocab = {token: idx for idx, token in enumerate(tokens)}
        self.vocab = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def mask_id(self) -> int:
        return self.vocab[self.mask_token]

    @property
    def cls_id(self) -> int:
        return self.vocab[self.cls_token]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def special_ids(self) -> set[int]:
        return {self.pad_id, self.mask_id, self.cls_id, self.unk_id}

    def __len__(self) -> int:
        return len(self.vocab)

    def tokenize(self, sequence: str) -> list[str]:
        sequence = sequence.upper()
        if len(sequence) < self.k:
            return []
        return [sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1)]

    def encode(
        self,
        sequence: str,
        *,
        add_cls: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        ids = [self.vocab.get(token, self.unk_id) for token in self.tokenize(sequence)]
        if add_cls:
            ids = [self.cls_id] + ids
        if max_length is not None:
            ids = ids[:max_length]
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> list[str]:
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if skip_special and token.startswith("[") and token.endswith("]"):
                continue
            tokens.append(token)
        return tokens

    def save(self, path: str | Path) -> None:
        payload = {"k": self.k, "vocab": self.vocab}
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KmerTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(k=int(payload["k"]), vocab=payload["vocab"])


def mask_tokens(
    input_ids: Sequence[int],
    *,
    mask_id: int,
    vocab_size: int,
    special_token_ids: Iterable[int],
    mask_probability: float = 0.15,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    """Apply BERT-style MLM masking.

    Labels are `-100` for unmasked positions, matching PyTorch's default
    ignore index for cross-entropy loss.
    """

    if not 0 < mask_probability < 1:
        raise ValueError("mask_probability must be between 0 and 1")
    rng = rng or random.Random()
    special = set(int(token_id) for token_id in special_token_ids)
    masked = list(int(token_id) for token_id in input_ids)
    labels = [-100] * len(masked)

    candidate_positions = [idx for idx, token_id in enumerate(masked) if token_id not in special]
    if not candidate_positions:
        return masked, labels

    n_to_mask = max(1, round(len(candidate_positions) * mask_probability))
    positions = rng.sample(candidate_positions, min(n_to_mask, len(candidate_positions)))

    for idx in positions:
        original_id = masked[idx]
        labels[idx] = original_id
        draw = rng.random()
        if draw < 0.8:
            masked[idx] = mask_id
        elif draw < 0.9:
            masked[idx] = rng.randrange(vocab_size)
        else:
            masked[idx] = original_id
    return masked, labels
