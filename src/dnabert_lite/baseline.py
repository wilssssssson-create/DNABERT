"""Traditional k-mer logistic regression baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import pr_auc_score_binary, roc_auc_score_binary, write_metrics, write_predictions
from .finetune import classification_metrics
from .tokenizer import KmerTokenizer


def read_labeled_sequences(path: str | Path, limit_samples: int | None = None) -> tuple[list[str], list[int]]:
    path = Path(path)
    sequences: list[str] = []
    labels: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "sequence" not in fieldnames or "label" not in fieldnames:
            raise ValueError(f"{path} must contain sequence and label columns")
        for row in reader:
            sequence = row["sequence"].strip().upper()
            label = row["label"].strip()
            if not sequence or label == "":
                continue
            sequences.append(sequence)
            labels.append(int(label))
            if limit_samples is not None and len(sequences) >= limit_samples:
                break
    if not sequences:
        raise ValueError(f"no labeled sequences found in {path}")
    return sequences, labels


def kmer_frequency_matrix(sequences: list[str], tokenizer: KmerTokenizer) -> np.ndarray:
    feature_size = len(tokenizer) - len(tokenizer.special_ids)
    features = np.zeros((len(sequences), feature_size), dtype=np.float32)
    offset = len(tokenizer.special_ids)

    for row_idx, sequence in enumerate(sequences):
        tokens = tokenizer.tokenize(sequence)
        total = 0
        for token in tokens:
            token_id = tokenizer.vocab.get(token)
            if token_id is None or token_id in tokenizer.special_ids:
                continue
            features[row_idx, token_id - offset] += 1.0
            total += 1
        if total > 0:
            features[row_idx] /= total
    return features


def metrics_from_scores(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    preds = [1 if score >= 0.5 else 0 for score in scores]
    logits = torch.tensor([[1.0 - score, score] for score in scores], dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    metrics: dict[str, float | None] = classification_metrics(logits, label_tensor)
    metrics["samples"] = float(len(labels))
    metrics["roc_auc"] = roc_auc_score_binary(labels, scores)
    metrics["pr_auc"] = pr_auc_score_binary(labels, scores)
    metrics["predicted_positive"] = float(sum(preds))
    return metrics


def train_kmer_baseline(args: argparse.Namespace) -> dict[str, object]:
    tokenizer = KmerTokenizer(k=args.k)
    train_sequences, train_labels = read_labeled_sequences(args.train_csv, args.limit_samples)
    val_sequences, val_labels = read_labeled_sequences(args.val_csv, args.limit_samples)
    test_sequences, test_labels = read_labeled_sequences(args.test_csv, args.limit_samples)

    x_train = kmer_frequency_matrix(train_sequences, tokenizer)
    x_val = kmer_frequency_matrix(val_sequences, tokenizer)
    x_test = kmer_frequency_matrix(test_sequences, tokenizer)

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=args.max_iter,
                    C=args.c,
                    class_weight=args.class_weight,
                    solver=args.solver,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(x_train, np.array(train_labels))

    val_scores = model.predict_proba(x_val)[:, 1].tolist()
    test_scores = model.predict_proba(x_test)[:, 1].tolist()
    val_preds = [1 if score >= 0.5 else 0 for score in val_scores]
    test_preds = [1 if score >= 0.5 else 0 for score in test_scores]

    val_metrics = metrics_from_scores(val_labels, val_scores)
    test_metrics = metrics_from_scores(test_labels, test_scores)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "k": args.k,
            "feature_type": "kmer_frequency",
            "vocab": tokenizer.vocab,
        },
        model_out,
    )

    summary: dict[str, object] = {
        "method": "kmer_logistic_regression",
        "k": args.k,
        "model_out": str(model_out),
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    flat_metrics: dict[str, float | None | str] = {
        "method": "kmer_logistic_regression",
        "k": float(args.k),
    }
    flat_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
    flat_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})

    if args.out:
        write_metrics(args.out, flat_metrics)
    if args.predictions_out:
        write_predictions(args.predictions_out, test_labels, test_scores, test_preds)
    return summary


def add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--out", default="results/kmer_baseline_metrics.csv")
    parser.add_argument("--predictions-out", default="results/kmer_baseline_predictions.csv")
    parser.add_argument("--model-out", default="checkpoints/kmer_logreg.joblib")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--class-weight", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit-samples", type=int, default=None)
