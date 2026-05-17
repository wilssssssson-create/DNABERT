import csv
from argparse import Namespace
from pathlib import Path

from dnabert_lite.evaluate import evaluate_checkpoint, pr_auc_score_binary, roc_auc_score_binary
from dnabert_lite.finetune import train_classifier
from tests.test_finetune import make_args, write_classification_csv


def test_auc_helpers():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]

    assert roc_auc_score_binary(labels, scores) == 0.75
    assert round(pr_auc_score_binary(labels, scores), 6) == 0.833333


def test_evaluate_checkpoint_writes_metrics_and_predictions(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    model_path = tmp_path / "classifier.pt"
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"
    write_classification_csv(train_csv)
    write_classification_csv(val_csv)
    train_classifier(make_args(train_csv, val_csv, model_path))

    summary = evaluate_checkpoint(
        Namespace(
            model=str(model_path),
            test_csv=str(val_csv),
            out=str(metrics_path),
            predictions_out=str(predictions_path),
            batch_size=2,
            max_length=None,
            device="cpu",
            limit_samples=None,
        )
    )

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert "roc_auc" in summary
    assert "pr_auc" in summary

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["samples"] == "4.0"
