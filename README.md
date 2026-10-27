# AI-Assisted Contract Review & Risk Triage

### Team Members

| Name | GitHub Handle | Main Contribution |
|---|---|---|
| [Name] | @[handle] | [Contribution] |
| [Name] | @[handle] | [Contribution] |
| [Name] | @[handle] | [Contribution] |
| [Name] | @[handle] | [Contribution] |
| [Name] | @[handle] | [Contribution] |
| [Name] | @[handle] | [Contribution] |
| [Name] | @[handle] | [Contribution] |

### Challenge Advisor
- [Name]
- [Role / Organization]

### AI Studio Coach
- [Name]

---

## Project Overview

This project was developed as part of the **Break Through Tech AI Studio Fall 2026** program in collaboration with **Accenture**.

The goal is to build an AI-assisted contract review pipeline that helps legal and procurement teams identify important contract clauses and prioritize contracts for further review.

Using the **Contract Understanding Atticus Dataset (CUAD)**, the project combines:

- exploratory data analysis and class imbalance analysis
- contract chunking and preprocessing
- multi-label clause classification
- TF-IDF and transformer-based modeling
- explainable rule-based risk scoring

The intended workflow is:

```text
CUAD Contracts
      ↓
EDA & Preprocessing
      ↓
Paragraph-Aware Chunking
      ↓
Multi-Label Clause Classification
TF-IDF Baseline → DistilRoBERTa
      ↓
Per-Category Precision / Recall / F1
      ↓
Clause-Level Risk Assessment
      ↓
Contract-Level Risk Triage
```

The system is intended to support human reviewers by helping surface potentially important clauses. It is not intended to replace legal review or provide legal conclusions.

---

## Dataset

We use the **CUAD dataset**, which contains:

- **510 commercial contracts**
- **41 expert-annotated contract clause categories**
- **13,823 annotated answer spans**

The project preserves CUAD's official contract-level train/test split:

- **408 training contracts**
- **102 test contracts**

Splitting at the contract level prevents chunks from the same contract from appearing in both training and testing.

Contract text is available directly in the JSON files through the `context` field, so raw PDFs do not need to be parsed.

### Data Files

Key data files are located under:

```text
data/cuad/
```

The modeling pipeline currently uses the official CUAD training and test JSON files.

---

## Category Scope

The project currently uses different category scopes for different stages of the pipeline.

### Risk Analysis Scope

The risk framework currently contains **19 selected CUAD categories** covering major contract-risk areas such as:

- liability
- assignment and control
- termination
- intellectual property
- licensing
- competition restrictions
- commercial obligations
- compliance

These categories are assigned preliminary **Low / Medium / High** risk levels in:

```text
evaluation/risk_rubric.md
```

### Baseline Comparison Scope

The initial TF-IDF baseline evaluates the following **7 categories**:

- Anti-Assignment
- Change Of Control
- Insurance
- Renewal Term
- Notice Period To Terminate Renewal
- Audit Rights
- Termination For Convenience

These categories serve as a consistent benchmark for comparing classical and transformer-based models.

### Transformer Modeling Scope

The DistilRoBERTa pipeline can include additional CUAD categories when they have sufficient positive training support.

The current dataset preparation script includes categories with at least **10 fully-contained positive training chunks**.

For an apples-to-apples comparison, transformer performance is also reported separately on the same 7 TF-IDF baseline categories.

---

## Exploratory Data Analysis

Exploratory analysis focused on understanding:

- class imbalance across CUAD categories
- positive vs. negative prevalence
- train/test category support
- contract and annotation length
- category overlap
- formatting and cleaning issues
- chunking requirements

The EDA confirmed substantial imbalance across categories.

Some categories appear in most contracts, while others appear in only a small number. Because of this imbalance, overall accuracy is not treated as the primary evaluation metric.

Instead, the project emphasizes:

- Precision
- Recall
- F1-score
- Per-category support

The EDA notebook is located at:

```text
notebooks/CUAD_EDA.ipynb
```

---

## Preprocessing & Chunking

Contracts can be much longer than the maximum sequence length supported by transformer models, so they must be divided into smaller sections.

The project uses a **paragraph-aware chunking strategy** based on:

- **480 tokens per chunk**
- **64-token overlap**
- the `distilroberta-base` tokenizer

The overlap helps preserve clauses that occur near chunk boundaries.

### Chunk Labeling

Each chunk-category pair is labeled using the CUAD answer spans:

- **Fully contained annotation → positive**
- **Partial-only overlap → masked**
- **No overlap → negative**

Partial-overlap cases are ignored during training rather than incorrectly being treated as negatives.

Dataset preparation is implemented in:

```text
training/prepare_dataset.py
```

Generated outputs include:

```text
train_chunks.csv
test_chunks.csv
train_tokenized.pt
test_tokenized.pt
labels.json
dataset_summary.csv
```

Generated data is excluded from Git tracking and can be regenerated from the scripts.

---

## TF-IDF Baseline

The initial machine-learning baseline uses:

- **TF-IDF features**
- unigram + bigram features
- **Logistic Regression**
- balanced class weighting
- CUAD's official train/test split

The baseline provides a simple reference point before using a transformer model.

### Baseline Results

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| Anti-Assignment | 0.807 | 0.931 | 0.865 |
| Audit Rights | 0.638 | 0.789 | 0.706 |
| Insurance | 0.583 | 0.656 | 0.618 |
| Termination For Convenience | 0.514 | 0.655 | 0.576 |
| Change Of Control | 0.517 | 0.577 | 0.545 |
| Notice Period To Terminate Renewal | 0.412 | 0.438 | 0.424 |
| Renewal Term | 0.393 | 0.423 | 0.407 |

The strongest baseline performance is on **Anti-Assignment**, while the renewal-related categories perform substantially worse.

One likely limitation is that whole-contract TF-IDF treats a long contract as a single bag of words, which can dilute short but important clauses.

The baseline notebook is located at:

```text
notebooks/naeem_tfidf_baseline_risk_rules.ipynb
```

---

## Transformer Model

The next modeling stage uses **DistilRoBERTa** for multi-label clause classification.

The training pipeline includes:

- `distilroberta-base`
- one output probability per clause category
- sigmoid multi-label classification
- masked binary cross-entropy loss
- contract-level train/validation splitting
- chunk-level and contract-level evaluation
- optional positive-class weighting

### Data Splits

The official CUAD train/test split is preserved.

Within the 408 official training contracts, approximately **10% of contracts are held out for validation**.

Validation is performed at the contract level so chunks from the same contract cannot appear in both the training and validation sets.

The official 102-contract test set remains separate for final evaluation.

Training code is located at:

```text
training/train_distilroberta.py
```

A Google Colab runner is available at:

```text
notebooks/naeem_distilroberta_finetune_colab.ipynb
```

---

## Class Imbalance Handling

CUAD contains significant imbalance both:

- **between categories**, where some labels have hundreds more positive examples than others
- **within categories**, where negative examples can significantly outnumber positives

The transformer training pipeline supports positive-class weighting using the ratio of negative to positive examples for each category.

Additional strategies being evaluated include:

- weighted loss
- positive-class weighting
- threshold tuning
- category-specific error analysis

---

## Risk Assessment Framework

The classification model identifies **what type of clause is present**.

A separate risk layer is responsible for estimating **how concerning that clause may be**.

The project currently uses three preliminary risk levels:

| Risk Level | Meaning |
|---|---|
| **Low** | Mostly routine or limited-consequence obligations |
| **Medium** | Meaningful restrictions or obligations whose impact depends on context |
| **High** | Potentially significant financial, legal, ownership, or strategic consequences |

The current framework includes **19 selected CUAD categories** with an initial risk assignment and justification.

Examples include:

- **Uncapped Liability → High**
- **Change Of Control → High**
- **IP Ownership Assignment → High**
- **Anti-Assignment → Medium**
- **License Grant → Medium**

The full rubric is available at:

```text
evaluation/risk_rubric.md
```

Structured category labels are available at:

```text
evaluation/risk_categories.csv
```

These labels are preliminary category-level assumptions, not ground-truth CUAD risk labels.

Actual clause-level risk may vary based on:

- financial exposure
- scope
- duration
- affected parties
- exceptions
- ownership consequences
- termination consequences
- business restrictions

---

## Repository Structure

```text
.
├── data/
│   └── cuad/
│
├── evaluation/
│   ├── risk_categories.csv
│   └── risk_rubric.md
│
├── notebooks/
│   ├── CUAD_EDA.ipynb
│   ├── JSONtoDF.ipynb
│   ├── naeem_tfidf_baseline_risk_rules.ipynb
│   └── naeem_distilroberta_finetune_colab.ipynb
│
├── training/
│   ├── prepare_dataset.py
│   ├── train_distilroberta.py
│   ├── requirements.txt
│   └── README.md
│
├── Challenge-Project-Overview.md
└── README.md
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Break-Through-Tech/Accenture-1F-contract-review-challenge.git
cd Accenture-1F-contract-review-challenge
```

Install training dependencies:

```bash
pip install -r training/requirements.txt
```

### Prepare the Dataset

```bash
python training/prepare_dataset.py
```

### Train DistilRoBERTa

```bash
python training/train_distilroberta.py --epochs 3
```

To test positive-class weighting:

```bash
python training/train_distilroberta.py --epochs 3 --use-pos-weight
```

For a quick smoke test:

```bash
python training/train_distilroberta.py --epochs 1 --limit 64
```

For GPU training, use the provided Google Colab notebook and select a **T4 GPU** runtime.

---

## Current Status

### Completed

- CUAD data exploration
- JSON → DataFrame conversion
- class imbalance analysis
- category-level EDA
- cleaning investigation
- chunking experiments
- TF-IDF + Logistic Regression baseline
- initial Low / Medium / High risk rubric
- paragraph-aware tokenization pipeline
- DistilRoBERTa training pipeline implementation

### In Progress

- generating and validating the final tokenized dataset
- running the initial DistilRoBERTa training baseline
- testing class-imbalance handling approaches
- comparing transformer performance against TF-IDF

### Next Steps

- compute and review per-category transformer metrics
- compare weighted vs. unweighted training
- conduct false-positive / false-negative error analysis
- refine risk rules using observed model failure cases
- build the contract-level risk-scoring layer
- validate risk rankings with advisor-reviewed examples
- integrate classification and risk scoring into an end-to-end pipeline

---

## Evaluation Strategy

Clause classification is evaluated primarily using:

- per-category Precision
- per-category Recall
- per-category F1
- micro F1
- macro F1

Both **chunk-level** and **contract-level** performance are considered.

Contract-level transformer scores are generated by aggregating predictions across a contract's chunks.

For the future risk-scoring layer, planned evaluation includes comparison against advisor-ranked examples using:

- ranking agreement
- bucket agreement
- sensitivity analysis around Low / Medium / High boundaries

---

## Limitations

- CUAD contains clause annotations but does not contain validated Low / Medium / High risk labels.
- Category prevalence is highly imbalanced.
- Some categories have very few positive examples.
- Risk depends heavily on contractual context and party perspective.
- Category-level risk assignments should not automatically determine the final risk of an individual clause.
- The system is designed for triage and prioritization, not autonomous legal decision-making.
- Human review remains necessary for consequential legal or business decisions.

---

## Project Goal

The long-term goal is an auditable contract-review pipeline that can:

1. identify important contractual clauses,
2. surface the relevant evidence,
3. estimate preliminary risk,
4. prioritize contracts or clauses for human review.
