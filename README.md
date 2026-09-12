# <a href="https://github.com/Anon-Author/X-LocVul">X-LocVul</a> Replication Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)

<br />
<p align="center">
  <h3 align="center">X-LocVul</h3>
  <p align="center">
    Auditing Evidence Propagation in Cascaded Detection, Localization, and Explanation of C/C++ Vulnerabilities
  </p>
</p>

## Overview

This repository contains the replication package for the **X-LocVul** paper. X-LocVul is a controlled empirical audit of a modular C/C++ pipeline:

1. **Stage 1**: UniXCoder-based function-level detector
2. **Stage 2**: CodeT5-based line generation with source-constrained projection
3. **Stage 3**: Qwen2.5-Coder explainer

<!-- Table of contents -->

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#directory-structure">Directory Structure</a></li>
    <li><a href="#mapping-to-manuscript-experiments">Mapping to Manuscript Experiments</a></li>
    <li><a href="#about-the-datasets">About the Datasets</a></li>
    <li><a href="#about-the-models">About the Models</a></li>
    <li><a href="#how-to-replicate-step-by-step-guide">How to Replicate (Step-by-Step Guide)</a>
      <ul>
        <li><a href="#0-prerequisites-and-environment-setup">0. Prerequisites and Environment Setup</a></li>
        <li><a href="#1-direct-inspection-of-results-fastest---no-execution-required">1. Direct Inspection of Results (Fastest)</a></li>
        <li><a href="#2-computational-reproduction-training--evaluation">2. Computational Reproduction (Training & Evaluation)</a></li>
      </ul>
    </li>
    <li><a href="#data-provenance-and-licenses">Data Provenance and Licenses</a></li>
    <li><a href="#known-limitations-and-reproducibility-boundaries">Known Limitations</a></li>
    <li><a href="#appendix-experimental-results">Appendix: Experimental Results</a></li>
  </ol>
</details>

## Directory Structure

To comply with GitHub storage constraints, large data and model weights are ignored via `.gitignore`. The core repository structure is:

```text
x-locvul/
├── README.md             
├── Makefile                  # Automated commands for reproduction
├── environment/              # Environment requirements
├── src/                      # Source code (Python scripts)
│   ├── e1/                   # Stage 1: Detection scripts
│   ├── e3/                   # Stage 2: Localization scripts
│   ├── e4/                   # Stage 3: Explanation scripts
│   ├── e5/                   # Chronological diagnostic scripts
│   └── e6/                   # Runtime cost scripts
├── results/                  # Retained raw logs and intermediate data
└── outputs/                  # Final generated CSV tables for the manuscript
```

## Mapping to Manuscript Experiments

Reviewers can trace conclusions from the paper directly to the exact data within minutes.

| RQ / Claim                                | Manuscript Section    | Evaluation Script (Reference)       | Pre-generated Output Table                    |
| :---------------------------------------- | :-------------------- | :---------------------------------- | :-------------------------------------------- |
| **RQ1**: Multi-task vs Single-task  | Section 5.1 & Table 1 | `src/e1/compute_e1_metrics.py`    | `outputs/e1/primevul_seed_metrics.csv`      |
| **RQ1**: Negative Control CI        | Section 5.1           | `src/e1/e1_bootstrap_all.py`      | `outputs/e1/e13_e15_bootstrap.csv`          |
| **RQ1**: Paired Tests               | Section 5.1           | `src/e1/compute_e1_metrics.py`    | `outputs/e1/paired_predictions.csv`         |
| **RQ2**: Fuzzy vs Semantic          | Section 5.2 & Table 3 | `src/e3/compute_e3_metrics.py`    | `outputs/e3/table_e3_final.csv`             |
| **RQ3**: Human Evaluation           | Section 5.3 & Table 4 | `src/e4/analyze_e4_ratings.py`    | `outputs/e4/e4_merged_scores.csv`           |
| **RQ3**: Holm Testing               | Section 5.3           | `src/e4/analyze_e4_ratings.py`    | `outputs/e4/e4_wilcoxon_tests.csv`          |
| **Diagnostic**: Chronological Shift | Section 5.1 & Table 2 | `src/e5/compute_e5_diagnostic.py` | `outputs/e5/per_seed_aggregate_metrics.csv` |
| **Runtime**: Cost & Filtering       | Section 6.4           | `src/e6/compute_e6_cost.py`       | `outputs/e6/runtime_summary.csv`            |

## About the Datasets

Our evaluation builds upon three well-known vulnerability datasets: **PrimeVul**, **BigVul**, and **LineVul**.

* **Stage 1 (Detection)** uses a merged and leakage-controlled Big-Vul/PrimeVul corpus (70/15/15 split). The test set contains 26,738 functions (24,816 clean and 1,922 vulnerable).
* **Stage 2 (Localization)** & **Stage 3 (Explanation)** utilize a quota-stratified sample of vulnerable LineVul functions balanced across CWE families.

> **Note on Data Availability:** The raw datasets are exceptionally large and require complex deduplication. **The fully pre-processed dataset used in our experiments will be uploaded to a Google Drive link shortly.**
>
> 📥 **[Download Processed Dataset Here](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

## About the Models

The X-LocVul pipeline cascades three specialized models:

1. **Stage 1 (Detection)**: Initialized from `UniXCoder-base` (and `CodeBERT-base` as a baseline). Trained for multi-task vulnerability and CWE detection on an NVIDIA Tesla T4 with a learning rate of `2e-5` and batch size of 8.
2. **Stage 2 (Localization)**: Utilizes `CodeT5-base` as a sequence-to-sequence generator for vulnerable statement projection. Trained with a learning rate of `5e-5` for 10 epochs using beam search (beam size 4).
3. **Stage 3 (Explanation)**: Uses `Qwen2.5-Coder-1.5B-Instruct` as a lightweight explainer. We employ a temperature of `0.7` and a 512-token output budget to generate root-cause analyses and repair suggestions.

## How to Replicate 

Because we do not provide pre-trained weights or cached prediction logs in this repository (due to size constraints), **evaluation scripts cannot be run in isolation**. You must either inspect the pre-generated results directly (Method 1) or run the full training pipeline before evaluating (Method 2).

### 0. Prerequisites and Environment Setup

* **OS**: Linux (Ubuntu 20.04+ recommended) or Windows with WSL2.
* **Hardware**: For full computational reproduction, an NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, A100) is strictly required.
* **Python**: Version 3.10 or higher.

**Step 1: Clone the repository**

```bash
git clone https://github.com/Anon-Author/X-LocVul.git
cd X-LocVul
```

**Step 2: Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

### 1. Direct Inspection of Results 

Because full reproduction requires days of GPU training, the fastest way to verify our claims is to inspect the pre-generated CSV tables located in the `outputs/` directory. We provide exhaustive tabular data covering all experimental stages:

#### E1: Stage-1 Detection

* `outputs/e1/primevul_seed_metrics.csv`: **[Corresponds to Table 1]** Main detection performance (F1, PR-AUC, etc.) on PrimeVul across E1.1-E1.6 configurations.
* `outputs/e1/bigvul_seed_metrics.csv`: Cross-project transfer performance on BigVul.
* `outputs/e1/e13_e15_bootstrap.csv`: Hierarchical-bootstrap CI comparing the Multi-task model against the Shuffled-CWE negative control.
* `outputs/e1/paired_predictions.csv`: McNemar contingency counts and Holm-adjusted p-values for paired significance testing.

#### E2: Auxiliary-Loss Sweep

* `outputs/e2/lambda_summary.csv`: Aggregated vulnerability F1, CWE macro-F1, and calibration scores across different λcwe values.
* `outputs/e2/per_seed_metrics.csv`: Raw metric outputs for each individual seed during the hyperparameter sweep.

#### E3: Stage-2 Generative Localization

* `outputs/e3/table_e3_final.csv`: **[Corresponds to Table 3]** Top-1/5/10 and MRR scores comparing raw generation, exact match, fuzzy projection, and semantic projection.
* `outputs/e3/e3_paired_comparison.csv`: Paired bootstrap comparisons between Fuzzy and Semantic projection strategies.

#### E4: Stage-3 Explanation Grounding (Human Evaluation)

* `outputs/e4/e4_merged_scores.csv`: **[Corresponds to Table 4]** Final human rating averages for root cause, evidence support, CWE consistency, repair usefulness, and unsupported claims.
* `outputs/e4/e4_wilcoxon_tests.csv`: Holm-corrected p-values from paired Wilcoxon tests across conditions.
* `outputs/e4/e4_condition_mean.csv`: Arithmetic mean scores for condition-level analyses.
* `outputs/e4/e4_irr_metrics.csv`: Inter-Rater Reliability (IRR) metrics including quadratic-weighted Cohen's Kappa.
* `outputs/e4/e4_rater_A.csv` & `e4_rater_B.csv`: Raw, blinded ratings before adjudication from the two independent human raters.

#### E5: Chronological Threshold-Transfer Diagnostic

* `outputs/e5/per_seed_aggregate_metrics.csv`: **[Corresponds to Table 2]** Chronological split metrics showing precision, recall, and F1 collapse at fixed validation thresholds.
* `outputs/e5/README_LIMITATION.md`: Notice detailing the boundaries of threshold recalibration due to storage constraints.

#### E6: Runtime and Cascade Cost

* `outputs/e6/runtime_summary.csv`: End-to-end average latency, peak memory profiles, and filtering cascade workloads for Stage 1/2/3.

### 2. Computational Reproduction (Training & Evaluation)

If you wish to rigorously reproduce the metrics computationally, you **must train the models from scratch first**, as weights and caches are not provided in this package.

**Step 1: Data Preparation**
Ensure you have downloaded the required dataset from Zenodo and placed the manifest files inside `data/manifests/`.

**Step 2: Train the Models**
To reproduce the training phase, execute the following commands sequentially for each experimental stage:

```bash
make data-prep   # Prepares, audits, and formats raw datasets
make train-e1    # Trains Stage 1 (Detection) models
make train-e3    # Trains Stage 2 (Localization) models
make train-e5    # Trains chronological diagnostic models
```

*(Note: Output weights will be saved to the `saved_models/` directory during this process).*

**Step 3: Evaluate and Generate Tables**
**Only after the corresponding training steps have finished**, run the individual evaluation targets to generate the CSV tables:

```bash
make e1          # Evaluates Stage 1 detection performance
make e1-bootstrap # Computes confidence intervals for E1
make e2          # Analyzes E2 auxiliary-loss MTL sweep
make e3          # Computes Top-K, MRR and ablations for localization
make e4          # Samples cases, evaluates, and analyzes E4 ratings
make e5          # Calculates chronological threshold diagnostic
make e6          # Summarizes runtime metrics
make run-pipeline # Runs the end-to-end multi-stage pipeline
```

Once finished, the tables in the `outputs/` directory will be overwritten with your newly reproduced data.

## Data Provenance and Licenses

* **Source Code**: Released under the MIT License.
* **Metadata, Tables, and Annotations**: CC BY 4.0.
* **Datasets**: PrimeVul, BigVul, and LineVul retain their original licenses.

## Known Limitations and Reproducibility Boundaries

As stipulated in our artifact design and Section 6.3 (Threats to Validity) of the paper:

* **E1**: Model weights (`.pt`) and cached prediction logs (`.npz`) are not provided due to storage constraints. Consequently, running evaluation without prior training will fail. We rely on the pre-generated tables in `outputs/` for immediate inspection.
* **E5**: The retained files contain seed-level aggregate metrics rather than sample-level probabilities due to storage limits. Consequently, the artifact reproduces the reported threshold-transfer diagnostic but cannot support post-hoc threshold recalibration.

## Appendix: Experimental Results

Below are the key results reported in the manuscript, directly reproducible via the pre-generated data tables in this repository

<div align="center">

<h3><b>Table 1: Controlled RQ1 comparison on the leakage-controlled PrimeVul-derived test set.</b></h3>

| Run | Configuration | Accuracy | MCC | Vul. F1 | PR-AUC | FPR |
|:---|:---|:---|:---|:---|:---|:---|
| E1.0a | Majority-clean | 0.9813 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1.0b | Lexical-logistic | 0.8249 | 0.1287 | 0.1015 | 0.0893 | 0.1696 |
| E1.1 | CodeBERT, single-task BCE | 0.9813 ± 0.0001 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1276 [0.0994, 0.1594] | 0.0006 ± 0.0006 |
| E1.2 | UniXCoder, single-task BCE (λcwe = 0) | 0.9812 ± 0.0003 | 0.1625 [0.1068, 0.2152] | 0.0829 [0.0488, 0.1199] | 0.1441 [0.1135, 0.1801] | 0.0011 ± 0.0009 |
| E1.3 | UniXCoder, multi-task, true CWE labels | 0.9813 ± 0.0001 | 0.1445 [0.0904, 0.1955] | 0.0773 [0.0449, 0.1121] | 0.1464 [0.1157, 0.1814] | 0.0011 ± 0.0006 |
| E1.5 | UniXCoder, multi-task, shuffled CWE (negative control) | 0.9814 ± 0.0001 | 0.1535 [0.0976, 0.2085] | 0.0820 [0.0471, 0.1191] | 0.1467 [0.1160, 0.1828] | 0.0009 ± 0.0002 |
| E1.6 | UniXCoder, multi-task, hierarchical (parent) CWE | 0.9815 ± 0.0000 | 0.0540 [-0.0013, 0.1078] | 0.0099 [0.0000, 0.0248] | 0.1427 [0.1140, 0.1785] | 0.0004 ± 0.0004 |
| E1.4 | UniXCoder, multi-task + hard-example mining | 0.9710 ± 0.0096 | 0.1569 [0.1160, 0.2031] | 0.1535 [0.1142, 0.1968] | 0.1128 [0.0912, 0.1392] | 0.0131 ± 0.0112 |

<br>

<h3><b>Table 2: Chronological/de-duplicated evaluation at the fixed validation threshold.</b></h3>

| Accuracy | MCC | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|:---|:---|:---|:---|:---|:---|:---|
| 0.9655 | −0.0018 | 0.0000 | 0.0000 | 0.0000 | 0.1097 | 0.7821 |

<br>

<h3><b>Table 3: Competing source-projection mechanisms in E3.</b></h3>

| Strategy | Top-1 | Top-5 | Top-10 | MRR |
|:---|:---|:---|:---|:---|
| Raw generation | 0.8306 [0.8053, 0.8559] | 0.8331 [0.8081, 0.8584] | 0.8351 [0.8101, 0.8602] | 0.8318 [0.8069, 0.8570] |
| Normalized exact match | 0.4160 [0.3817, 0.4470] | 0.4173 [0.3830, 0.4480] | 0.4173 [0.3830, 0.4480] | 0.4166 [0.3825, 0.4475] |
| Fuzzy projection | 0.8379 [0.8127, 0.8630] | 0.8580 [0.8353, 0.8817] | 0.8583 [0.8353, 0.8820] | 0.8465 [0.8225, 0.8711] |
| Semantic projection | 0.8221 [0.7957, 0.8487] | 0.8490 [0.8259, 0.8739] | 0.8493 [0.8264, 0.8741] | 0.8331 [0.8097, 0.8591] |

<br>

<h3><b>Table 4: Two-rater E4 results over 50 paired cases.</b></h3>

| Condition | Root cause | Evidence | CWE | Repair | Unsupported claim |
|:---|:---|:---|:---|:---|:---|
| Function only | 1.40 | 2.27 | 1.40 | 1.33 | 0.90 |
| Function + line | 1.34 | 3.42 | 1.72 | 1.33 | 0.90 |
| Function + line + CWE | 1.40 | 3.51 | 1.40 | 1.27 | 0.86 |
| Oracle evidence | 1.86 | 4.43 | 3.70 | 1.59 | 0.87 |

</div>
