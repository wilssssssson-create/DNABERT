from argparse import Namespace
from pathlib import Path

from dnabert_lite.preprocess import IndexedFasta, preprocess


def test_indexed_fasta_fetch(tmp_path: Path):
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">chr1\nACGTAC\nGTACGT\n>chr2\nTTTTAA\n", encoding="utf-8")

    genome = IndexedFasta(fasta)

    assert genome.fetch("chr1", 0, 4) == "ACGT"
    assert genome.fetch("chr1", 4, 10) == "ACGTAC"
    assert genome.fetch("chr2", 2, 6) == "TTAA"


def test_preprocess_tiny_dataset(tmp_path: Path):
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(
        ">chr1\n" + "ACGT" * 100 + "\n"
        ">chr17\n" + "TGCA" * 100 + "\n"
        ">chr19\n" + "CAGT" * 100 + "\n",
        encoding="utf-8",
    )
    promoter_bed = tmp_path / "promoter.bed"
    promoter_bed.write_text(
        "chr1\t40\t80\tp1\nchr17\t40\t80\tp2\nchr19\t40\t80\tp3\n",
        encoding="utf-8",
    )
    enhancer_bed = tmp_path / "enhancer.bed"
    enhancer_bed.write_text("", encoding="utf-8")

    out_dir = tmp_path / "processed"
    summary = preprocess(
        Namespace(
            genome=str(fasta),
            promoter_bed=str(promoter_bed),
            enhancer_bed=str(enhancer_bed),
            fantom5_bed=None,
            out_dir=str(out_dir),
            positive_source="promoter",
            include_fantom5=False,
            seq_length=20,
            max_per_split=1,
            negatives_per_positive=1,
            pretrain_samples=2,
            seed=1,
        )
    )

    assert summary["splits"]["train"]["positive"] == 1
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "pretrain.csv").exists()
