"""Command line interface for DNABERT-lite."""

from __future__ import annotations

import argparse
import json

from .baseline import add_baseline_args, train_kmer_baseline
from .evaluate import add_evaluate_args, evaluate_checkpoint
from .finetune import add_finetune_args, train_classifier
from .pretrain import add_pretrain_args, train_mlm
from .preprocess import add_preprocess_args, preprocess
from .visualize import (
    add_embed_args,
    add_visualize_embedding_args,
    extract_embeddings,
    visualize_embeddings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dnabert-lite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Build CSV datasets from FASTA and BED files.")
    add_preprocess_args(preprocess_parser)

    pretrain_parser = subparsers.add_parser("pretrain", help="Run masked language modeling pretraining.")
    add_pretrain_args(pretrain_parser)

    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a sequence classifier.")
    add_finetune_args(finetune_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a classifier checkpoint.")
    add_evaluate_args(evaluate_parser)

    embed_parser = subparsers.add_parser("embed", help="Extract sequence embeddings from a classifier checkpoint.")
    add_embed_args(embed_parser)

    visualize_parser = subparsers.add_parser("visualize-embedding", help="Run PCA and plot embeddings.")
    add_visualize_embedding_args(visualize_parser)

    baseline_parser = subparsers.add_parser("kmer-baseline", help="Train a k-mer logistic regression baseline.")
    add_baseline_args(baseline_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        summary = preprocess(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "pretrain":
        summary = train_mlm(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "finetune":
        summary = train_classifier(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "evaluate":
        summary = evaluate_checkpoint(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "embed":
        summary = extract_embeddings(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "visualize-embedding":
        summary = visualize_embeddings(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "kmer-baseline":
        summary = train_kmer_baseline(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
