import csv
from argparse import Namespace
from pathlib import Path

import numpy as np

from dnabert_lite.baseline import kmer_frequency_matrix, train_kmer_baseline
from dnabert_lite.tokenizer import KmerTokenizer
from tests.test_finetune import write_classification_csv


def test_kmer_frequency_matrix_rows_sum_to_one():
    tokenizer = KmerTokenizer(k=3)
    features = kmer_frequency_matrix(["ACGTAC"], tokenizer)

    assert features.shape == (1, 64)
    assert np.isclose(features.sum(), 1.0)


def test_train_kmer_baseline_outputs_metrics(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"
    model_path = tmp_path / "baseline.joblib"
    write_classification_csv(train_csv)
    write_classification_csv(val_csv)
    write_classification_csv(test_csv)

    summary = train_kmer_baseline(
        Namespace(
            train_csv=str(train_csv),
            val_csv=str(val_csv),
            test_csv=str(test_csv),
            out=str(metrics_path),
            predictions_out=str(predictions_path),
            model_out=str(model_path),
            k=3,
            c=1.0,
            max_iter=200,
            solver="lbfgs",
            class_weight=None,
            seed=1,
            limit_samples=None,
        )
    )

    assert summary["train_samples"] == 4
    assert "roc_auc" in summary["test_metrics"]
    assert metrics_path.exists()
    assert predictions_path.exists()
    assert model_path.exists()

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["method"] == "kmer_logistic_regression"
