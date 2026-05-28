# DNABERT-lite

A teaching-oriented DNABERT-style encoder project for DNA k-mer representation
learning and promoter/enhancer classification.

The project implements the main pieces needed for a compact DNABERT-like
pipeline: k-mer tokenization, positional encoding, multi-head self-attention,
BERT-style encoder blocks, masked language modeling pretraining, downstream
classification, k-mer logistic regression baseline, evaluation, and embedding
PCA visualization.

## Current status

The repository contains runnable code, tests, and experiment scripts. Large raw
genome/regulatory-element files are intentionally not committed to git, so a
fresh clone needs local raw data placed under `data/raw/` before preprocessing.
Some processed data and experiment outputs may exist in a local working copy,
but they should be treated as generated artifacts rather than source files.

## Data layout

Expected local raw data files:

```text
data/raw/hg38/hg38.fa
data/raw/encode_screen/GRCh38-cCREs.PLS.bed
data/raw/encode_screen/GRCh38-cCREs.ELS.bed
data/raw/fantom5/F5.hg38.enhancers.bed.gz
```

These files are too large for normal git tracking. Download them from the
corresponding public sources, or copy them from local storage, then keep the same
relative paths. The preprocessing step will generate smaller CSV files such as
`train.csv`, `val.csv`, `test.csv`, and `pretrain.csv`.

## Install

From this directory:

```bash
python3 -m pip install -e '.[ml]'
```

If you do not want to install yet, run commands with:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli --help
```

## Preprocess data

Default small promoter-vs-background dataset:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli preprocess \
  --genome data/raw/hg38/hg38.fa \
  --promoter-bed data/raw/encode_screen/GRCh38-cCREs.PLS.bed \
  --enhancer-bed data/raw/encode_screen/GRCh38-cCREs.ELS.bed \
  --out-dir data/processed \
  --positive-source promoter \
  --seq-length 200 \
  --max-per-split 1000 \
  --pretrain-samples 3000
```

Outputs:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/pretrain.csv
data/processed/preprocess_summary.json
```

## MLM pretraining

After preprocessing, run a small masked language modeling experiment:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli pretrain \
  --train-csv data/processed/pretrain.csv \
  --out checkpoints/mlm_pretrained.pt \
  --k 6 \
  --max-length 256 \
  --epochs 3 \
  --batch-size 32 \
  --hidden-size 128 \
  --num-layers 2 \
  --num-heads 4 \
  --intermediate-size 256
```

For a very quick CPU smoke test:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli pretrain \
  --train-csv data/processed/pretrain.csv \
  --out checkpoints/mlm_smoke.pt \
  --epochs 1 \
  --batch-size 2 \
  --hidden-size 16 \
  --num-layers 1 \
  --num-heads 4 \
  --intermediate-size 32 \
  --max-length 64 \
  --device cpu \
  --limit-samples 8
```

## Fine-tuning

Random-initialized DNABERT-lite classifier baseline:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli finetune \
  --train-csv data/processed/train.csv \
  --val-csv data/processed/val.csv \
  --test-csv data/processed/test.csv \
  --out checkpoints/classifier_random.pt \
  --k 6 \
  --max-length 256 \
  --epochs 5 \
  --batch-size 32
```

Fine-tune from the MLM-pretrained encoder:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli finetune \
  --train-csv data/processed/train.csv \
  --val-csv data/processed/val.csv \
  --test-csv data/processed/test.csv \
  --pretrained checkpoints/mlm_pretrained.pt \
  --out checkpoints/classifier_pretrained.pt \
  --max-length 256 \
  --epochs 5 \
  --batch-size 32
```

## Evaluation

Evaluate a saved classifier checkpoint and write metrics for experiment tables:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli evaluate \
  --model checkpoints/classifier_pretrained.pt \
  --test-csv data/processed/test.csv \
  --out results/pretrained_metrics.csv \
  --predictions-out results/pretrained_predictions.csv \
  --batch-size 32
```

The metrics include accuracy, precision, recall, F1, confusion counts,
ROC-AUC, and PR-AUC.

## k-mer Logistic Regression Baseline

Train and evaluate the traditional k-mer frequency baseline:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli kmer-baseline \
  --train-csv data/processed/train.csv \
  --val-csv data/processed/val.csv \
  --test-csv data/processed/test.csv \
  --out results/kmer_baseline_metrics.csv \
  --predictions-out results/kmer_baseline_predictions.csv \
  --model-out checkpoints/kmer_logreg.joblib \
  --k 6
```

This produces metrics compatible with the random and pretrained DNABERT-lite
classifier outputs, so the three methods can be compared in one table.

## Reproducible Experiment Script

Run the full comparison pipeline with GPU-aware pretraining parameters:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py --dry-run
```

Remove `--dry-run` to execute the pipeline:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --out-dir experiments/promoter_auto
```

The script detects GPU memory and selects one of these profiles:

| Profile | Typical hardware | Hidden | Layers | Heads | Max length | Batch | Pretrain samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu | no GPU | 16 | 1 | 4 | 64 | 4 | 500 |
| tiny | <8GB GPU | 32 | 1 | 4 | 128 | 8 | 1000 |
| small | 8GB GPU | 64 | 2 | 4 | 128 | 16 | 3000 |
| medium | 12-16GB GPU | 128 | 2 | 4 | 256 | 32 | 6000 |
| large | 24-48GB GPU | 192 | 4 | 6 | 256 | 48 | 12000 |
| xlarge | A100/A800/H100 or 60GB+ | 256 | 6 | 8 | 384 | 64 | 24000 |

For a faster course-project run on a large GPU, start with:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile medium \
  --out-dir experiments/promoter_medium
```

For the course report, see [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md). It gives
a recommended experiment plan, including the main promoter-vs-background
experiment, pretraining comparison, k-mer/sequence-length ablations, optional
enhancer/regulatory extensions, and copy-paste runnable commands.

## Embedding PCA

Extract sequence-level embeddings from a classifier checkpoint:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli embed \
  --model checkpoints/classifier_pretrained.pt \
  --input-csv data/processed/test.csv \
  --out results/pretrained_embeddings.npy \
  --metadata-out results/pretrained_embedding_metadata.csv \
  --batch-size 32
```

Run PCA and save a coordinate table plus a figure:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli visualize-embedding \
  --embedding results/pretrained_embeddings.npy \
  --metadata results/pretrained_embedding_metadata.csv \
  --coords-out results/pretrained_embedding_pca.csv \
  --out results/pretrained_embedding_pca.png \
  --label-column label_name
```

Try enhancer classification instead:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli preprocess \
  --positive-source enhancer \
  --seq-length 200 \
  --max-per-split 1000
```

Use both promoter-like and enhancer-like regions as positives:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli preprocess \
  --positive-source regulatory \
  --seq-length 200 \
  --max-per-split 1000
```

Add FANTOM5 enhancers to the positive set:

```bash
PYTHONPATH=src python3 -m dnabert_lite.cli preprocess \
  --positive-source regulatory \
  --include-fantom5
```

## Run tests

```bash
PYTHONPATH=src pytest tests
```
