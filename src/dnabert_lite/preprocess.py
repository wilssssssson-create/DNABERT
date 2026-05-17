"""Preprocessing utilities for raw hg38 FASTA and regulatory BED files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


DEFAULT_SPLITS = {
    "train": tuple(f"chr{i}" for i in range(1, 17)),
    "val": ("chr17", "chr18"),
    "test": ("chr19", "chr20", "chr21", "chr22"),
}


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int
    source: str


@dataclass(frozen=True)
class FastaIndexEntry:
    chrom: str
    length: int
    offset: int
    line_bases: int
    line_width: int


class IndexedFasta:
    """Random-access FASTA reader for standard line-wrapped genome FASTA files."""

    def __init__(self, fasta_path: str | Path) -> None:
        self.path = Path(fasta_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.index = self._build_index()

    def _build_index(self) -> dict[str, FastaIndexEntry]:
        index: dict[str, FastaIndexEntry] = {}
        current_chrom: str | None = None
        current_offset = 0
        current_length = 0
        line_bases = 0
        line_width = 0

        with self.path.open("rb") as handle:
            while True:
                raw = handle.readline()
                if not raw:
                    break
                if raw.startswith(b">"):
                    if current_chrom is not None:
                        index[current_chrom] = FastaIndexEntry(
                            current_chrom,
                            current_length,
                            current_offset,
                            line_bases,
                            line_width,
                        )
                    header = raw[1:].decode("ascii", errors="ignore").strip()
                    current_chrom = header.split()[0]
                    current_offset = handle.tell()
                    current_length = 0
                    line_bases = 0
                    line_width = 0
                    continue

                stripped = raw.rstrip(b"\r\n")
                if not stripped:
                    continue
                if line_bases == 0:
                    line_bases = len(stripped)
                    line_width = len(raw)
                current_length += len(stripped)

        if current_chrom is not None:
            index[current_chrom] = FastaIndexEntry(
                current_chrom,
                current_length,
                current_offset,
                line_bases,
                line_width,
            )
        return index

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if chrom not in self.index:
            raise KeyError(f"chromosome not found in FASTA: {chrom}")
        entry = self.index[chrom]
        start = max(0, int(start))
        end = min(int(end), entry.length)
        if start >= end:
            return ""

        row_start = start // entry.line_bases
        col_start = start % entry.line_bases
        byte_offset = entry.offset + row_start * entry.line_width + col_start
        bases_needed = end - start
        extra_newlines = ((col_start + bases_needed) // entry.line_bases + 2) * (
            entry.line_width - entry.line_bases
        )
        read_size = bases_needed + max(0, extra_newlines)

        with self.path.open("rb") as handle:
            handle.seek(byte_offset)
            raw = handle.read(read_size)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"")[:bases_needed]
        return seq.decode("ascii").upper()


def open_text(path: str | Path) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def clean_sequence(sequence: str) -> str:
    return sequence.upper()


def is_clean_dna(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= {"A", "C", "G", "T"}


def read_bed(path: str | Path, source: str) -> list[Region]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    regions: list[Region] = []
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if chrom.startswith("chr") and end > start:
                regions.append(Region(chrom, start, end, source))
    return regions


def centered_window(region: Region, seq_length: int, chrom_length: int) -> tuple[int, int]:
    center = (region.start + region.end) // 2
    start = center - seq_length // 2
    start = max(0, min(start, chrom_length - seq_length))
    end = start + seq_length
    return start, end


def split_for_chrom(chrom: str, splits: dict[str, tuple[str, ...]]) -> str | None:
    for split, chroms in splits.items():
        if chrom in chroms:
            return split
    return None


def merge_intervals(regions: Iterable[Region]) -> dict[str, tuple[list[int], list[int]]]:
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for region in regions:
        by_chrom.setdefault(region.chrom, []).append((region.start, region.end))

    merged: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        chrom_merged: list[list[int]] = []
        for start, end in intervals:
            if not chrom_merged or start > chrom_merged[-1][1]:
                chrom_merged.append([start, end])
            else:
                chrom_merged[-1][1] = max(chrom_merged[-1][1], end)
        starts = [item[0] for item in chrom_merged]
        ends = [item[1] for item in chrom_merged]
        merged[chrom] = (starts, ends)
    return merged


def overlaps_merged(
    chrom: str,
    start: int,
    end: int,
    merged: dict[str, tuple[list[int], list[int]]],
) -> bool:
    if chrom not in merged:
        return False
    starts, ends = merged[chrom]
    previous = bisect_right(starts, start) - 1
    if previous >= 0 and ends[previous] > start:
        return True
    next_idx = previous + 1
    return next_idx < len(starts) and starts[next_idx] < end


def sample_background(
    fasta: IndexedFasta,
    chroms: tuple[str, ...],
    seq_length: int,
    count: int,
    rng: random.Random,
    occupied: dict[str, tuple[list[int], list[int]]],
    source: str,
    max_attempts_per_sample: int = 1000,
) -> list[dict[str, object]]:
    available_chroms = [chrom for chrom in chroms if chrom in fasta.index and fasta.index[chrom].length > seq_length]
    weights = [fasta.index[chrom].length - seq_length for chrom in available_chroms]
    rows: list[dict[str, object]] = []
    attempts = 0
    max_attempts = max_attempts_per_sample * max(1, count)

    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        chrom = rng.choices(available_chroms, weights=weights, k=1)[0]
        chrom_length = fasta.index[chrom].length
        start = rng.randint(0, chrom_length - seq_length)
        end = start + seq_length
        if overlaps_merged(chrom, start, end, occupied):
            continue
        sequence = clean_sequence(fasta.fetch(chrom, start, end))
        if not is_clean_dna(sequence):
            continue
        rows.append(
            {
                "sequence": sequence,
                "label": 0,
                "label_name": "background",
                "source": source,
                "chrom": chrom,
                "start": start,
                "end": end,
            }
        )
    return rows


def collect_positive_rows(
    fasta: IndexedFasta,
    regions: list[Region],
    seq_length: int,
    max_per_split: int,
    rng: random.Random,
    splits: dict[str, tuple[str, ...]],
) -> dict[str, list[dict[str, object]]]:
    shuffled = list(regions)
    rng.shuffle(shuffled)
    rows_by_split = {split: [] for split in splits}

    for region in shuffled:
        split = split_for_chrom(region.chrom, splits)
        if split is None or len(rows_by_split[split]) >= max_per_split:
            continue
        if region.chrom not in fasta.index or fasta.index[region.chrom].length < seq_length:
            continue
        start, end = centered_window(region, seq_length, fasta.index[region.chrom].length)
        sequence = clean_sequence(fasta.fetch(region.chrom, start, end))
        if not is_clean_dna(sequence):
            continue
        rows_by_split[split].append(
            {
                "sequence": sequence,
                "label": 1,
                "label_name": region.source,
                "source": region.source,
                "chrom": region.chrom,
                "start": start,
                "end": end,
            }
        )
    return rows_by_split


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "sequence", "label", "label_name", "source", "chrom", "start", "end", "split"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            output = {field: row.get(field, "") for field in fieldnames}
            output["id"] = f"{path.stem}_{idx:06d}"
            writer.writerow(output)


def build_positive_regions(args: argparse.Namespace) -> list[Region]:
    regions: list[Region] = []
    if args.positive_source in {"promoter", "regulatory"}:
        regions.extend(read_bed(args.promoter_bed, "promoter"))
    if args.positive_source in {"enhancer", "regulatory"}:
        regions.extend(read_bed(args.enhancer_bed, "enhancer"))
    if args.include_fantom5:
        if args.fantom5_bed is None:
            raise ValueError("--include-fantom5 requires --fantom5-bed")
        regions.extend(read_bed(args.fantom5_bed, "fantom5_enhancer"))
    return regions


def preprocess(args: argparse.Namespace) -> dict[str, object]:
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta = IndexedFasta(args.genome)
    positive_regions = build_positive_regions(args)
    occupied = merge_intervals(positive_regions)

    positive_by_split = collect_positive_rows(
        fasta,
        positive_regions,
        args.seq_length,
        args.max_per_split,
        rng,
        DEFAULT_SPLITS,
    )

    summary: dict[str, object] = {
        "positive_source": args.positive_source,
        "seq_length": args.seq_length,
        "seed": args.seed,
        "raw_positive_regions": len(positive_regions),
        "splits": {},
    }

    for split, positive_rows in positive_by_split.items():
        negative_count = len(positive_rows) * args.negatives_per_positive
        negative_rows = sample_background(
            fasta,
            DEFAULT_SPLITS[split],
            args.seq_length,
            negative_count,
            rng,
            occupied,
            source=f"random_background_{split}",
        )
        rows = positive_rows + negative_rows
        rng.shuffle(rows)
        for row in rows:
            row["split"] = split
        write_csv(out_dir / f"{split}.csv", rows)
        summary["splits"][split] = {
            "positive": len(positive_rows),
            "background": len(negative_rows),
            "total": len(rows),
        }

    pretrain_rows = sample_background(
        fasta,
        DEFAULT_SPLITS["train"],
        args.seq_length,
        args.pretrain_samples,
        rng,
        occupied,
        source="random_genome_pretrain",
    )
    for row in pretrain_rows:
        row["label"] = ""
        row["label_name"] = ""
        row["split"] = "train"
    write_csv(out_dir / "pretrain.csv", pretrain_rows)
    summary["pretrain_samples"] = len(pretrain_rows)

    summary_path = out_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def add_preprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--genome", default="data/raw/hg38/hg38.fa")
    parser.add_argument("--promoter-bed", default="data/raw/encode_screen/GRCh38-cCREs.PLS.bed")
    parser.add_argument("--enhancer-bed", default="data/raw/encode_screen/GRCh38-cCREs.ELS.bed")
    parser.add_argument("--fantom5-bed", default="data/raw/fantom5/F5.hg38.enhancers.bed.gz")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--positive-source",
        choices=("promoter", "enhancer", "regulatory"),
        default="promoter",
    )
    parser.add_argument("--include-fantom5", action="store_true")
    parser.add_argument("--seq-length", type=int, default=200)
    parser.add_argument("--max-per-split", type=int, default=1000)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--pretrain-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
