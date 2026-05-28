# Experiment Guide

This guide is designed for the course report. The central question is:

> Can a lightweight DNABERT-style encoder learn useful DNA sequence
> representations through masked language modeling, and does that help
> promoter/enhancer classification compared with random initialization and a
> traditional k-mer frequency baseline?

## 1. Prepare the Environment

Run all commands from the repository root:

```bash
cd /export/home/sunyiming/course/bio2502_DNABERT
python3 -m pip install -e '.[ml]'
```

The raw data is not committed because the genome FASTA and BED files are large.
Before running preprocessing, make sure these files exist locally:

```text
data/raw/hg38/hg38.fa
data/raw/encode_screen/GRCh38-cCREs.PLS.bed
data/raw/encode_screen/GRCh38-cCREs.ELS.bed
data/raw/fantom5/F5.hg38.enhancers.bed.gz
```

Use a dry run first to print the exact commands without launching training:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile large \
  --out-dir experiments/report_promoter_main \
  --positive-source promoter \
  --seq-length 200 \
  --k 6 \
  --max-per-split 10000 \
  --pretrain-samples 50000 \
  --dry-run
```

## 2. Recommended Report Experiments

Use these as the main experimental sections:

| Section | Experiment | Purpose |
| --- | --- | --- |
| Main comparison | k-mer logistic regression vs. random DNABERT-lite vs. pretrained DNABERT-lite | Shows whether the neural encoder and MLM pretraining help |
| Pretraining effect | random encoder vs. MLM-pretrained encoder under the same data/model setting | Directly answers the project question |
| k-mer ablation | k=5 vs. k=6, sequence length fixed at 200 | Tests tokenizer granularity |
| sequence-length ablation | length=200 vs. length=300, k fixed at 6 | Tests whether longer context helps |
| embedding visualization | PCA of random vs. pretrained sequence embeddings | Provides qualitative representation analysis |
| optional extension | enhancer or regulatory vs. background | Shows the pipeline generalizes beyond promoters |

For a concise report, the strongest core set is:

```text
experiments/report_promoter_main
experiments/report_k5_len200
experiments/report_k6_len300
```

## 3. Main Promoter Experiment

This runs preprocessing, k-mer logistic regression, MLM pretraining, fine-tuning,
evaluation, embedding extraction, and PCA visualization.

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile large \
  --out-dir experiments/report_promoter_main \
  --positive-source promoter \
  --seq-length 200 \
  --k 6 \
  --max-per-split 10000 \
  --pretrain-samples 50000
```

Use the generated files in the report:

```text
experiments/report_promoter_main/results/kmer_baseline_metrics.csv
experiments/report_promoter_main/results/random_metrics.csv
experiments/report_promoter_main/results/pretrained_metrics.csv
experiments/report_promoter_main/results/random_embedding_pca.png
experiments/report_promoter_main/results/pretrained_embedding_pca.png
experiments/report_promoter_main/processed/preprocess_summary.json
```

Report metrics: accuracy, precision, recall, F1, ROC-AUC, and PR-AUC. F1 and
ROC-AUC are the most useful headline metrics.

## 4. k-mer Ablation

Run the same promoter task with k=5 and sequence length 200:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile large \
  --out-dir experiments/report_k5_len200 \
  --positive-source promoter \
  --seq-length 200 \
  --k 5 \
  --max-per-split 10000 \
  --pretrain-samples 50000
```

Compare it with the main k=6 result:

```text
experiments/report_k5_len200/results/pretrained_metrics.csv
experiments/report_promoter_main/results/pretrained_metrics.csv
```

In the report, discuss whether shorter k-mers improve recall by being less
sparse, or whether longer k-mers improve specificity by carrying more local
sequence context.

## 5. Sequence-Length Ablation

Run k=6 with longer 300 bp input sequences:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile large \
  --out-dir experiments/report_k6_len300 \
  --positive-source promoter \
  --seq-length 300 \
  --k 6 \
  --max-per-split 10000 \
  --pretrain-samples 50000
```

Compare it with the main length=200 result:

```text
experiments/report_k6_len300/results/pretrained_metrics.csv
experiments/report_promoter_main/results/pretrained_metrics.csv
```

This section answers whether the model benefits from more genomic context.

## 6. Optional Enhancer or Regulatory Experiment

If time allows, run one extra task to show the same pipeline works for another
biological label.

Enhancer vs. background:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile medium \
  --out-dir experiments/report_enhancer_k6_len200 \
  --positive-source enhancer \
  --seq-length 200 \
  --k 6 \
  --max-per-split 5000 \
  --pretrain-samples 20000
```

Promoter/enhancer regulatory regions vs. background:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile medium \
  --out-dir experiments/report_regulatory_k6_len200 \
  --positive-source regulatory \
  --seq-length 200 \
  --k 6 \
  --max-per-split 5000 \
  --pretrain-samples 20000
```

Only include this section if the core promoter experiments finish first.

## 7. Faster Debug Commands

Use these when checking that the pipeline works before committing GPU time.

CPU smoke test:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile cpu \
  --out-dir experiments/debug_cpu \
  --positive-source promoter \
  --seq-length 200 \
  --k 6 \
  --dry-run
```

Small real run:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --profile small \
  --out-dir experiments/debug_small \
  --positive-source promoter \
  --seq-length 200 \
  --k 6
```

## 8. Suggested Report Structure

1. Biological question: promoter/enhancer classification from DNA sequence.
2. Method: k-mer tokenizer, BERT-style encoder, MLM objective, classifier head.
3. Dataset: hg38 sequence extraction, ENCODE cCRE promoter/enhancer labels,
   random genomic background, train/val/test sizes from `preprocess_summary.json`.
4. Experiments: baseline comparison, pretraining effect, k-mer ablation,
   sequence-length ablation, embedding PCA.
5. Results: table of metrics and PCA figures.
6. Discussion: whether MLM pretraining improves F1/ROC-AUC, when k-mer logistic
   regression is competitive, and what the embedding visualization suggests.

## 9. Existing Local Results

Your current working tree already contains completed promoter experiments under
`experiments/`, including:

```text
experiments/promoter_large_more_data
experiments/final_promoter_k5_len200
experiments/final_promoter_k6_len200
experiments/final_promoter_k6_len300
```

These are useful for drafting the report before rerunning everything. For
example, `experiments/promoter_large_more_data/results/` contains model metrics,
prediction CSVs, PCA files, and several ready-to-use PNG/PDF figures.
