"""Run a reproducible DNABERT-lite experiment pipeline.

The script detects the available GPU model/memory and selects a conservative
model/data profile for MLM pretraining. Use --dry-run first to inspect commands.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    max_per_split: int
    pretrain_samples: int
    max_length: int
    pretrain_epochs: int
    finetune_epochs: int
    batch_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    device: str
    pretrain_lr: float
    finetune_lr: float


PROFILES = {
    "cpu": ExperimentProfile("cpu", 100, 500, 64, 1, 2, 4, 16, 1, 4, 32, "cpu", 1e-5, 1e-5),
    "tiny": ExperimentProfile("tiny", 300, 1000, 128, 1, 3, 8, 32, 1, 4, 64, "auto", 1e-5, 1e-5),
    "small": ExperimentProfile("small", 1000, 3000, 128, 3, 5, 16, 64, 2, 4, 128, "auto", 1e-5, 1e-5),
    "medium": ExperimentProfile("medium", 2000, 6000, 256, 5, 6, 32, 128, 2, 4, 256, "auto", 1e-5, 1e-5),
    "large": ExperimentProfile("large", 5000, 12000, 256, 8, 8, 48, 192, 4, 6, 384, "auto", 1e-5, 1e-5),
    "xlarge": ExperimentProfile("xlarge", 10000, 24000, 384, 10, 10, 64, 256, 6, 8, 512, "auto", 1e-5, 1e-5),
}


def detect_gpus() -> list[tuple[str, int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, memory = line.rsplit(",", 1)
        gpus.append((name.strip(), int(memory.strip())))
    return gpus


def choose_profile(gpus: list[tuple[str, int]]) -> str:
    if not gpus:
        return "cpu"

    name = max(gpus, key=lambda item: item[1])[0].lower()
    memory_mb = max(memory for _, memory in gpus)
    if any(token in name for token in ("h100", "a100", "a800", "80gb")) or memory_mb >= 60000:
        return "xlarge"
    if memory_mb >= 24000:
        return "large"
    if memory_mb >= 12000:
        return "medium"
    if memory_mb >= 8000:
        return "small"
    return "tiny"


def build_command(repo_root: Path, *args: str) -> list[str]:
    return [sys.executable, "-m", "dnabert_lite.cli", *args]


def run_command(command: list[str], repo_root: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}")
    if dry_run:
        return

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def selected_steps(args: argparse.Namespace) -> set[str]:
    if not args.steps:
        return {"preprocess", "baseline", "pretrain", "finetune", "evaluate", "embed", "visualize"}
    return set(args.steps.split(","))


def run_experiment(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gpus = detect_gpus()
    auto_profile = choose_profile(gpus)
    profile_name = args.profile if args.profile != "auto" else auto_profile
    profile = PROFILES[profile_name]
    steps = selected_steps(args)

    print(f"Detected GPUs: {gpus if gpus else 'none'}")
    print(f"Selected profile: {profile.name}")
    print(f"Outputs: {args.out_dir}")

    processed_dir = Path(args.out_dir) / "processed"
    checkpoint_dir = Path(args.out_dir) / "checkpoints"
    result_dir = Path(args.out_dir) / "results"

    if "preprocess" in steps:
        run_command(
            build_command(
                repo_root,
                "preprocess",
                "--genome",
                args.genome,
                "--promoter-bed",
                args.promoter_bed,
                "--enhancer-bed",
                args.enhancer_bed,
                "--fantom5-bed",
                args.fantom5_bed,
                "--out-dir",
                str(processed_dir),
                "--positive-source",
                args.positive_source,
                "--seq-length",
                str(args.seq_length),
                "--max-per-split",
                str(args.max_per_split or profile.max_per_split),
                "--pretrain-samples",
                str(args.pretrain_samples or profile.pretrain_samples),
                "--seed",
                str(args.seed),
            ),
            repo_root,
            args.dry_run,
        )

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"
    pretrain_csv = processed_dir / "pretrain.csv"

    if "baseline" in steps:
        run_command(
            build_command(
                repo_root,
                "kmer-baseline",
                "--train-csv",
                str(train_csv),
                "--val-csv",
                str(val_csv),
                "--test-csv",
                str(test_csv),
                "--out",
                str(result_dir / "kmer_baseline_metrics.csv"),
                "--predictions-out",
                str(result_dir / "kmer_baseline_predictions.csv"),
                "--model-out",
                str(checkpoint_dir / "kmer_logreg.joblib"),
                "--k",
                str(args.k),
            ),
            repo_root,
            args.dry_run,
        )

    mlm_checkpoint = checkpoint_dir / "mlm_pretrained.pt"
    if "pretrain" in steps:
        run_command(
            build_command(
                repo_root,
                "pretrain",
                "--train-csv",
                str(pretrain_csv),
                "--out",
                str(mlm_checkpoint),
                "--k",
                str(args.k),
                "--max-length",
                str(profile.max_length),
                "--epochs",
                str(args.pretrain_epochs or profile.pretrain_epochs),
                "--batch-size",
                str(args.batch_size or profile.batch_size),
                "--hidden-size",
                str(profile.hidden_size),
                "--num-layers",
                str(profile.num_layers),
                "--num-heads",
                str(profile.num_heads),
                "--intermediate-size",
                str(profile.intermediate_size),
                "--learning-rate",
                str(args.pretrain_lr or profile.pretrain_lr),
                "--device",
                profile.device,
                "--seed",
                str(args.seed),
            ),
            repo_root,
            args.dry_run,
        )

    random_checkpoint = checkpoint_dir / "classifier_random.pt"
    pretrained_checkpoint = checkpoint_dir / "classifier_pretrained.pt"
    if "finetune" in steps:
        shared = [
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--test-csv",
            str(test_csv),
            "--max-length",
            str(profile.max_length),
            "--epochs",
            str(args.finetune_epochs or profile.finetune_epochs),
            "--batch-size",
            str(args.batch_size or profile.batch_size),
            "--device",
            profile.device,
            "--learning-rate",
            str(args.finetune_lr or profile.finetune_lr),
            "--best-metric",
            args.best_metric,
            "--seed",
            str(args.seed),
        ]
        run_command(
            build_command(
                repo_root,
                "finetune",
                "--out",
                str(random_checkpoint),
                "--k",
                str(args.k),
                "--hidden-size",
                str(profile.hidden_size),
                "--num-layers",
                str(profile.num_layers),
                "--num-heads",
                str(profile.num_heads),
                "--intermediate-size",
                str(profile.intermediate_size),
                *shared,
            ),
            repo_root,
            args.dry_run,
        )
        run_command(
            build_command(
                repo_root,
                "finetune",
                "--pretrained",
                str(mlm_checkpoint),
                "--out",
                str(pretrained_checkpoint),
                *shared,
            ),
            repo_root,
            args.dry_run,
        )

    if "evaluate" in steps:
        for label, checkpoint in (
            ("random", random_checkpoint),
            ("pretrained", pretrained_checkpoint),
        ):
            run_command(
                build_command(
                    repo_root,
                    "evaluate",
                    "--model",
                    str(checkpoint),
                    "--test-csv",
                    str(test_csv),
                    "--out",
                    str(result_dir / f"{label}_metrics.csv"),
                    "--predictions-out",
                    str(result_dir / f"{label}_predictions.csv"),
                    "--batch-size",
                    str(args.batch_size or profile.batch_size),
                    "--device",
                    profile.device,
                ),
                repo_root,
                args.dry_run,
            )

    if "embed" in steps:
        for label, checkpoint in (
            ("random", random_checkpoint),
            ("pretrained", pretrained_checkpoint),
        ):
            run_command(
                build_command(
                    repo_root,
                    "embed",
                    "--model",
                    str(checkpoint),
                    "--input-csv",
                    str(test_csv),
                    "--out",
                    str(result_dir / f"{label}_embeddings.npy"),
                    "--metadata-out",
                    str(result_dir / f"{label}_embedding_metadata.csv"),
                    "--batch-size",
                    str(args.batch_size or profile.batch_size),
                    "--device",
                    profile.device,
                ),
                repo_root,
                args.dry_run,
            )

    if "visualize" in steps:
        for label in ("random", "pretrained"):
            run_command(
                build_command(
                    repo_root,
                    "visualize-embedding",
                    "--embedding",
                    str(result_dir / f"{label}_embeddings.npy"),
                    "--metadata",
                    str(result_dir / f"{label}_embedding_metadata.csv"),
                    "--coords-out",
                    str(result_dir / f"{label}_embedding_pca.csv"),
                    "--out",
                    str(result_dir / f"{label}_embedding_pca.png"),
                    "--label-column",
                    "label_name",
                ),
                repo_root,
                args.dry_run,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["auto", *PROFILES.keys()], default="auto")
    parser.add_argument("--steps", default=None, help="Comma-separated subset, e.g. preprocess,pretrain,finetune")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default="experiments/promoter_auto")
    parser.add_argument("--positive-source", choices=("promoter", "enhancer", "regulatory"), default="promoter")
    parser.add_argument("--genome", default="data/raw/hg38/hg38.fa")
    parser.add_argument("--promoter-bed", default="data/raw/encode_screen/GRCh38-cCREs.PLS.bed")
    parser.add_argument("--enhancer-bed", default="data/raw/encode_screen/GRCh38-cCREs.ELS.bed")
    parser.add_argument("--fantom5-bed", default="data/raw/fantom5/F5.hg38.enhancers.bed.gz")
    parser.add_argument("--seq-length", type=int, default=200)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-per-split", type=int, default=None)
    parser.add_argument("--pretrain-samples", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--best-metric", choices=("f1", "accuracy", "precision", "recall"), default="f1")
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
