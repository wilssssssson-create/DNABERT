"""Embedding extraction and PCA visualization for DNABERT-lite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import load_classifier_checkpoint
from .pretrain import resolve_device


def read_sequence_rows(path: str | Path, limit_samples: int | None = None) -> list[dict[str, str]]:
    path = Path(path)
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "sequence" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a sequence column")
        for row in reader:
            if row.get("sequence", "").strip():
                rows.append(dict(row))
            if limit_samples is not None and len(rows) >= limit_samples:
                break
    if not rows:
        raise ValueError(f"no sequences found in {path}")
    return rows


def write_metadata(path: str | Path, rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = ["embedding_index"]
    for row in rows:
        for key in row:
            if key != "sequence" and key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for idx, row in enumerate(rows):
            output = {"embedding_index": idx}
            output.update({key: value for key, value in row.items() if key != "sequence"})
            writer.writerow(output)


def extract_embeddings(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    model, tokenizer = load_classifier_checkpoint(args.model)
    model.to(device)
    model.eval()

    max_length = args.max_length or model.encoder.config.max_position_embeddings
    rows = read_sequence_rows(args.input_csv, args.limit_samples)
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            input_ids = [
                tokenizer.encode(row["sequence"], add_cls=True, max_length=max_length)
                for row in batch_rows
            ]
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            attention_mask = input_tensor.ne(tokenizer.pad_id)
            output = model.encoder(input_tensor, attention_mask=attention_mask)
            embeddings.append(output["pooler_output"].detach().cpu().numpy())

    embedding_array = np.concatenate(embeddings, axis=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embedding_array)

    metadata_out = Path(args.metadata_out)
    write_metadata(metadata_out, rows)

    summary = {
        "model": str(args.model),
        "input_csv": str(args.input_csv),
        "embedding_out": str(out_path),
        "metadata_out": str(metadata_out),
        "num_sequences": int(embedding_array.shape[0]),
        "embedding_dim": int(embedding_array.shape[1]),
    }
    return summary


def pca_2d(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if embeddings.shape[0] < 2:
        raise ValueError("at least two embeddings are required for PCA")

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variances = singular_values**2 / max(1, embeddings.shape[0] - 1)
    total_variance = variances.sum()
    explained = variances[:2] / total_variance if total_variance > 0 else np.zeros(2)

    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        explained = np.array([explained[0], 0.0])
    return coords, explained


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_pca_coords(
    path: str | Path,
    coords: np.ndarray,
    metadata_rows: list[dict[str, str]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_keys: list[str] = []
    for row in metadata_rows:
        for key in row:
            if key not in metadata_keys:
                metadata_keys.append(key)
    fieldnames = ["embedding_index", "pc1", "pc2"] + [
        key for key in metadata_keys if key != "embedding_index"
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, point in enumerate(coords):
            row = {"embedding_index": idx, "pc1": point[0], "pc2": point[1]}
            if idx < len(metadata_rows):
                row.update({key: value for key, value in metadata_rows[idx].items() if key != "embedding_index"})
            writer.writerow(row)


def plot_pca(
    path: str | Path,
    coords: np.ndarray,
    metadata_rows: list[dict[str, str]],
    label_column: str,
    explained_variance: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        row.get(label_column) or row.get("label_name") or row.get("label") or "unknown"
        for row in metadata_rows
    ]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, label in enumerate(unique_labels):
        mask = np.array([item == label for item in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=24,
            alpha=0.8,
            color=cmap(idx % 10),
            label=label,
        )

    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def visualize_embeddings(args: argparse.Namespace) -> dict[str, object]:
    embeddings = np.load(args.embedding)
    metadata_rows = read_metadata(args.metadata)
    if len(metadata_rows) != embeddings.shape[0]:
        raise ValueError("metadata row count must match number of embeddings")

    coords, explained = pca_2d(embeddings)
    coords_out = Path(args.coords_out)
    write_pca_coords(coords_out, coords, metadata_rows)

    if args.out:
        plot_pca(args.out, coords, metadata_rows, args.label_column, explained)

    summary = {
        "embedding": str(args.embedding),
        "metadata": str(args.metadata),
        "coords_out": str(coords_out),
        "plot_out": str(args.out) if args.out else None,
        "num_sequences": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "pc1_explained_variance": float(explained[0]),
        "pc2_explained_variance": float(explained[1]),
    }
    return summary


def add_embed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-csv", default="data/processed/test.csv")
    parser.add_argument("--out", default="results/embeddings.npy")
    parser.add_argument("--metadata-out", default="results/embedding_metadata.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int, default=None)


def add_visualize_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding", default="results/embeddings.npy")
    parser.add_argument("--metadata", default="results/embedding_metadata.csv")
    parser.add_argument("--coords-out", default="results/embedding_pca.csv")
    parser.add_argument("--out", default="results/embedding_pca.png")
    parser.add_argument("--label-column", default="label_name")
