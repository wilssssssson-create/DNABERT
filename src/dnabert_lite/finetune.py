"""Sequence classification fine-tuning for DNABERT-lite."""

from __future__ import annotations

import argparse
import csv
import json
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .model import DNABertLiteConfig, DNABertLiteForSequenceClassification
from .pretrain import resolve_device, set_seed
from .tokenizer import KmerTokenizer
from .training_utils import ProgressBar, parallelize_model, scalar_loss, unwrap_model


class CsvClassificationDataset(Dataset):
    """Read DNA sequences and integer labels from a CSV file."""

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
        self.rows = self._read_rows(limit_samples)
        if not self.rows:
            raise ValueError(f"no labeled sequences found in {self.csv_path}")

    def _read_rows(self, limit_samples: int | None) -> list[tuple[str, int]]:
        rows: list[tuple[str, int]] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "sequence" not in fieldnames or "label" not in fieldnames:
                raise ValueError(f"{self.csv_path} must contain sequence and label columns")
            for row in reader:
                sequence = row["sequence"].strip().upper()
                label = row["label"].strip()
                if not sequence or label == "":
                    continue
                rows.append((sequence, int(label)))
                if limit_samples is not None and len(rows) >= limit_samples:
                    break
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        sequence, label = self.rows[index]
        input_ids = self.tokenizer.encode(sequence, add_cls=True, max_length=self.max_length)
        return input_ids, label


class ClassificationCollator:
    """Build tensors for sequence classification batches."""

    def __init__(self, tokenizer: KmerTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: Sequence[tuple[Sequence[int], int]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([example[0] for example in examples], dtype=torch.long)
        labels = torch.tensor([example[1] for example in examples], dtype=torch.long)
        attention_mask = input_ids.ne(self.tokenizer.pad_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = torch.argmax(logits, dim=-1)
    labels = labels.to(preds.device)

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    total = max(1, tp + tn + fp + fn)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = scalar_loss(output["loss"])
            total_loss += float(loss.detach().cpu())
            steps += 1
            all_logits.append(output["logits"].detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = classification_metrics(logits, labels)
    metrics["loss"] = total_loss / max(1, steps)
    metrics["steps"] = float(steps)
    metrics["samples"] = float(labels.numel())
    model.train()
    return metrics


def load_pretrained_checkpoint(path: str | Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_tokenizer_and_config(args: argparse.Namespace) -> tuple[KmerTokenizer, DNABertLiteConfig]:
    if args.pretrained:
        checkpoint = load_pretrained_checkpoint(args.pretrained)
        tokenizer_info = checkpoint.get("tokenizer", {})
        config_info = dict(checkpoint["config"])
        tokenizer = KmerTokenizer(k=int(tokenizer_info.get("k", args.k)), vocab=tokenizer_info.get("vocab"))
        config_info["num_labels"] = args.num_labels
        config = DNABertLiteConfig(**config_info)
        if args.max_length > config.max_position_embeddings:
            raise ValueError("--max-length cannot exceed pretrained max_position_embeddings")
        return tokenizer, config

    tokenizer = KmerTokenizer(k=args.k)
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
        num_labels=args.num_labels,
    )
    return tokenizer, config


def train_classifier(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer, config = build_tokenizer_and_config(args)

    train_dataset = CsvClassificationDataset(
        args.train_csv,
        tokenizer,
        max_length=args.max_length,
        limit_samples=args.limit_samples,
    )
    val_dataset = CsvClassificationDataset(
        args.val_csv,
        tokenizer,
        max_length=args.max_length,
        limit_samples=args.limit_samples,
    )
    test_dataset = None
    if args.test_csv:
        test_dataset = CsvClassificationDataset(
            args.test_csv,
            tokenizer,
            max_length=args.max_length,
            limit_samples=args.limit_samples,
        )

    collator = ClassificationCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )

    model = DNABertLiteForSequenceClassification(config)
    if args.pretrained:
        checkpoint = load_pretrained_checkpoint(args.pretrained)
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model, gpu_count = parallelize_model(
        model,
        device,
        use_multi_gpu=getattr(args, "multi_gpu", True),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, object]] = []
    best_metric = float("-inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] | None = None

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_steps = 0
        progress = ProgressBar(
            len(train_loader),
            f"Fine-tune epoch {epoch}/{args.epochs}",
            enabled=getattr(args, "progress", True),
        )
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = scalar_loss(output["loss"])
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_steps += 1
            progress.update(loss=total_loss / total_steps)
        progress.close()

        train_loss = total_loss / max(1, total_steps)
        val_metrics = evaluate_classifier(model, val_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_steps": total_steps,
            "val": val_metrics,
        }
        history.append(epoch_summary)
        if val_metrics[args.best_metric] > best_metric:
            best_metric = val_metrics[args.best_metric]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())
        print(json.dumps(epoch_summary, sort_keys=True))

    if best_state_dict is not None:
        unwrap_model(model).load_state_dict(best_state_dict)
    final_val_metrics = evaluate_classifier(model, val_loader, device)
    final_test_metrics = evaluate_classifier(model, test_loader, device) if test_loader is not None else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_model = unwrap_model(model)
    checkpoint = {
        "model_state_dict": checkpoint_model.state_dict(),
        "encoder_state_dict": checkpoint_model.encoder.state_dict(),
        "config": asdict(config),
        "tokenizer": {"k": tokenizer.k, "vocab": tokenizer.vocab},
        "history": history,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "best_metric": args.best_metric,
        "best_metric_value": best_metric,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "pretrained": args.pretrained,
        "args": vars(args),
    }
    torch.save(checkpoint, out_path)

    summary = {
        "checkpoint": str(out_path),
        "device": str(device),
        "gpu_count": gpu_count,
        "pretrained": args.pretrained,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else 0,
        "epochs": args.epochs,
        "best_metric": args.best_metric,
        "best_metric_value": best_metric,
        "best_epoch": best_epoch,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
    }
    return summary


def add_finetune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--out", default="checkpoints/classifier.pt")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--best-metric", choices=("f1", "accuracy", "precision", "recall"), default="f1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit-samples", type=int, default=None)
