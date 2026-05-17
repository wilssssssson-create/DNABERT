import csv
from argparse import Namespace
from pathlib import Path

import numpy as np

from dnabert_lite.finetune import train_classifier
from dnabert_lite.visualize import extract_embeddings, pca_2d, visualize_embeddings
from tests.test_finetune import make_args, write_classification_csv


def test_extract_embeddings_and_pca_outputs(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    model_path = tmp_path / "classifier.pt"
    embedding_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.csv"
    coords_path = tmp_path / "coords.csv"
    write_classification_csv(train_csv)
    write_classification_csv(val_csv)
    train_classifier(make_args(train_csv, val_csv, model_path))

    embed_summary = extract_embeddings(
        Namespace(
            model=str(model_path),
            input_csv=str(val_csv),
            out=str(embedding_path),
            metadata_out=str(metadata_path),
            batch_size=2,
            max_length=None,
            device="cpu",
            limit_samples=None,
        )
    )

    embeddings = np.load(embedding_path)
    assert embeddings.shape == (4, 16)
    assert embed_summary["num_sequences"] == 4
    assert metadata_path.exists()

    viz_summary = visualize_embeddings(
        Namespace(
            embedding=str(embedding_path),
            metadata=str(metadata_path),
            coords_out=str(coords_path),
            out=None,
            label_column="label",
        )
    )

    assert coords_path.exists()
    assert viz_summary["embedding_dim"] == 16
    with coords_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "pc1" in rows[0]
    assert "pc2" in rows[0]


def test_pca_2d_shape():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    coords, explained = pca_2d(embeddings)

    assert coords.shape == (3, 2)
    assert explained.shape == (2,)
