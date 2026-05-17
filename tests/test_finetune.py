import csv
from argparse import Namespace
from pathlib import Path

import torch

from dnabert_lite.finetune import ClassificationCollator, CsvClassificationDataset, train_classifier
from dnabert_lite.tokenizer import KmerTokenizer


def write_classification_csv(path: Path) -> None:
    rows = [
        ("ACGTACGTACGTACGT", 1),
        ("TGCATGCATGCATGCA", 1),
        ("CCCCAAAAGGGGTTTT", 0),
        ("AAAACCCCGGGGTTTT", 0),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "label"])
        writer.writeheader()
        for sequence, label in rows:
            writer.writerow({"sequence": sequence, "label": label})


def make_args(train_csv: Path, val_csv: Path, out_path: Path, pretrained: str | None = None) -> Namespace:
    return Namespace(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        test_csv=str(val_csv),
        pretrained=pretrained,
        out=str(out_path),
        k=3,
        max_length=12,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        dropout=0.0,
        num_labels=2,
        max_grad_norm=1.0,
        best_metric="f1",
        device="cpu",
        seed=1,
        limit_samples=None,
    )


def test_classification_collator_shapes(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    write_classification_csv(csv_path)
    tokenizer = KmerTokenizer(k=3)
    dataset = CsvClassificationDataset(csv_path, tokenizer, max_length=12)

    batch = ClassificationCollator(tokenizer)([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"].shape == (2, 12)
    assert batch["labels"].tolist() == [1, 1]


def test_train_classifier_saves_checkpoint(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    out_path = tmp_path / "classifier.pt"
    write_classification_csv(train_csv)
    write_classification_csv(val_csv)

    summary = train_classifier(make_args(train_csv, val_csv, out_path))

    checkpoint = torch.load(out_path, map_location="cpu", weights_only=False)
    assert summary["train_samples"] == 4
    assert out_path.exists()
    assert "model_state_dict" in checkpoint
    assert "val_metrics" in checkpoint
    assert checkpoint["best_metric"] == "f1"
    assert checkpoint["best_epoch"] == 1
    assert "f1" in checkpoint["val_metrics"]
