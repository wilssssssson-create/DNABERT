"""Standalone evaluation for DNABERT-lite classification checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .finetune import ClassificationCollator, CsvClassificationDataset, classification_metrics
from .model import DNABertLiteConfig, DNABertLiteForSequenceClassification
from .pretrain import resolve_device
from .training_utils import scalar_loss
from .tokenizer import KmerTokenizer


def roc_auc_score_binary(labels: list[int], scores: list[float]) -> float | None:
    """Compute binary ROC-AUC with average ranks for tied scores."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and scores[order[end]] == scores[order[idx]]:
            end += 1
        average_rank = (idx + 1 + end) / 2
        for rank_idx in range(idx, end):
            ranks[order[rank_idx]] = average_rank
        idx = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def pr_auc_score_binary(labels: list[int], scores: list[float]) -> float | None:
    """Compute average precision, commonly reported as PR-AUC for binary tasks."""

    positives = sum(labels)
    if positives == 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def load_classifier_checkpoint(path: str | Path) -> tuple[DNABertLiteForSequenceClassification, KmerTokenizer]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer_info = checkpoint["tokenizer"]
    tokenizer = KmerTokenizer(k=int(tokenizer_info["k"]), vocab=tokenizer_info["vocab"])
    config = DNABertLiteConfig(**checkpoint["config"])
    model = DNABertLiteForSequenceClassification(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer


def collect_predictions(
    model: DNABertLiteForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = scalar_loss(output["loss"])
            total_loss += float(loss.detach().cpu())
            steps += 1
            all_logits.append(output["logits"].detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), total_loss, steps


def write_metrics(path: str | Path, metrics: dict[str, float | None | str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def write_predictions(
    path: str | Path,
    labels: list[int],
    scores: list[float],
    preds: list[int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "score", "prediction"])
        writer.writeheader()
        for idx, (label, score, pred) in enumerate(zip(labels, scores, preds)):
            writer.writerow({"id": idx, "label": label, "score": score, "prediction": pred})


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, float | None | str]:
    device = resolve_device(args.device)
    model, tokenizer = load_classifier_checkpoint(args.model)
    model.to(device)

    max_length = args.max_length or model.encoder.config.max_position_embeddings
    dataset = CsvClassificationDataset(
        args.test_csv,
        tokenizer,
        max_length=max_length,
        limit_samples=args.limit_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ClassificationCollator(tokenizer),
    )
    logits, label_tensor, total_loss, steps = collect_predictions(model, dataloader, device)
    probabilities = torch.softmax(logits, dim=-1)
    positive_scores = probabilities[:, 1].tolist()
    labels = [int(label) for label in label_tensor.tolist()]
    preds = torch.argmax(logits, dim=-1).tolist()

    metrics = classification_metrics(logits, label_tensor)
    metrics["loss"] = total_loss / max(1, steps)
    metrics["steps"] = float(steps)
    metrics["samples"] = float(len(labels))
    metrics["roc_auc"] = roc_auc_score_binary(labels, positive_scores)
    metrics["pr_auc"] = pr_auc_score_binary(labels, positive_scores)

    summary: dict[str, float | None | str] = {
        "model": str(args.model),
        "test_csv": str(args.test_csv),
        **metrics,
    }

    if args.out:
        write_metrics(args.out, summary)
    if args.predictions_out:
        write_predictions(args.predictions_out, labels, positive_scores, [int(pred) for pred in preds])
    return summary


def add_evaluate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--out", default="results/metrics.csv")
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None)
