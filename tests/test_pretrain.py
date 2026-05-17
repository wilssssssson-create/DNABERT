import csv
from argparse import Namespace
from pathlib import Path

import torch

from dnabert_lite.pretrain import CsvSequenceDataset, MlmCollator, train_mlm
from dnabert_lite.tokenizer import KmerTokenizer


def write_pretrain_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence"])
        writer.writeheader()
        writer.writerow({"sequence": "ACGTACGTACGTACGT"})
        writer.writerow({"sequence": "TGCATGCATGCATGCA"})
        writer.writerow({"sequence": "CCCCAAAAGGGGTTTT"})
        writer.writerow({"sequence": "AAAACCCCGGGGTTTT"})


def test_mlm_collator_shapes(tmp_path: Path):
    csv_path = tmp_path / "pretrain.csv"
    write_pretrain_csv(csv_path)
    tokenizer = KmerTokenizer(k=3)
    dataset = CsvSequenceDataset(csv_path, tokenizer, max_length=12)
    batch = MlmCollator(tokenizer, seed=1)([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"].shape == (2, 12)
    assert batch["labels"].shape == (2, 12)
    assert batch["labels"][:, 0].tolist() == [-100, -100]


def test_train_mlm_saves_checkpoint(tmp_path: Path):
    csv_path = tmp_path / "pretrain.csv"
    out_path = tmp_path / "mlm.pt"
    write_pretrain_csv(csv_path)

    summary = train_mlm(
        Namespace(
            train_csv=str(csv_path),
            out=str(out_path),
            k=3,
            max_length=12,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            mask_probability=0.15,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            intermediate_size=32,
            dropout=0.0,
            max_grad_norm=1.0,
            device="cpu",
            seed=1,
            limit_samples=None,
        )
    )

    checkpoint = torch.load(out_path, map_location="cpu", weights_only=False)
    assert summary["num_sequences"] == 4
    assert out_path.exists()
    assert "model_state_dict" in checkpoint
    assert checkpoint["config"]["hidden_size"] == 16
