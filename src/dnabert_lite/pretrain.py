"""Masked language modeling pretraining for DNABERT-lite."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .model import DNABertLiteConfig, DNABertLiteForMaskedLM
from .tokenizer import KmerTokenizer, mask_tokens


class CsvSequenceDataset(Dataset):
    """Read DNA sequences from a CSV file with a `sequence` column."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: KmerTokenizer,
        max_length: int,
        limit_samples: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = self._read_sequences(limit_samples)
        if not self.sequences:
            raise ValueError(f"no sequences found in {self.csv_path}")

    def _read_sequences(self, limit_samples: int | None) -> list[str]:
        sequences: list[str] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "sequence" not in (reader.fieldnames or []):
                raise ValueError(f"{self.csv_path} must contain a sequence column")
            for row in reader:
                sequence = row["sequence"].strip().upper()
                if sequence:
                    sequences.append(sequence)
                if limit_samples is not None and len(sequences) >= limit_samples:
                    break
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.tokenizer.encode(
            self.sequences[index],
            add_cls=True,
            max_length=self.max_length,
        )


class MlmCollator:
    """Create masked inputs and MLM labels for a batch of encoded sequences."""

    def __init__(
        self,
        tokenizer: KmerTokenizer,
        mask_probability: float = 0.15,
        seed: int = 13,
    ) -> None:
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.rng = random.Random(seed)

    def __call__(self, examples: Sequence[Sequence[int]]) -> dict[str, torch.Tensor]:
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        for ids in examples:
            masked_ids, label_ids = mask_tokens(
                ids,
                mask_id=self.tokenizer.mask_id,
                vocab_size=len(self.tokenizer),
                special_token_ids=self.tokenizer.special_ids,
                mask_probability=self.mask_probability,
                rng=self.rng,
            )
            input_ids.append(masked_ids)
            labels.append(label_ids)

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = input_tensor.ne(self.tokenizer.pad_id)
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_mlm(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = KmerTokenizer(k=args.k)
    dataset = CsvSequenceDataset(
        args.train_csv,
        tokenizer,
        max_length=args.max_length,
        limit_samples=args.limit_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=MlmCollator(tokenizer, args.mask_probability, args.seed),
    )

    config = DNABertLiteConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.max_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_dropout_prob=args.dropout,
        pad_token_id=tokenizer.pad_id,
    )
    model = DNABertLiteForMaskedLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_steps = 0
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_steps += 1

        average_loss = total_loss / max(1, total_steps)
        epoch_summary = {"epoch": epoch, "steps": total_steps, "loss": average_loss}
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, sort_keys=True))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "config": asdict(config),
        "tokenizer": {"k": tokenizer.k, "vocab": tokenizer.vocab},
        "history": history,
        "args": vars(args),
    }
    torch.save(checkpoint, out_path)

    summary = {
        "checkpoint": str(out_path),
        "device": str(device),
        "num_sequences": len(dataset),
        "epochs": args.epochs,
        "final_loss": history[-1]["loss"] if history else None,
    }
    return summary


def add_pretrain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-csv", default="data/processed/pretrain.csv")
    parser.add_argument("--out", default="checkpoints/mlm_pretrained.pt")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit-samples", type=int, default=None)
