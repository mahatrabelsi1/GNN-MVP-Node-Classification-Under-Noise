# DiaGraph - Robust Diabetes Prediction on Noisy Graphs

DiaGraph is a prediction-only GitHub mini-competition for graph machine learning.

It is not an application.  
It is a benchmark and evaluation pipeline where participants train models locally and submit only prediction files via Pull Requests.

---

## What This Project Is

This project is:

- Supervised learning (train/val labels are provided)
- Graph machine learning task
- Node classification problem  
  (each node = one patient, predict diabetes label per node)
- A robustness-oriented benchmark  
  (class imbalance, graph sparsity, missingness indicators, hidden test labels)

One-line description:

> A supervised graph node classification benchmark for robust diabetes prediction under noisy conditions.

---

## Objective

Predict whether a node represents a diabetic patient (binary classification) using:

- Node features (clinical attributes)
- Graph edges / adjacency (similarity relations between patients)

Both GNN models and non-GNN baselines are allowed.

---

## Graph Specification (Mandatory)

- Node feature matrix X: `data/public/nodes.csv`
  - Column `id` is the node id (0-based).
  - All other columns are node features.
- Adjacency matrix A: `data/public/edges.csv`
  - Each row is a directed edge (`src`, `dst`) using the same node ids.

Dataset sizes (current release):
- Nodes: 96,128
- Edges: 1,224,756
- Graph density (directed): ~0.013%

---

## Dataset Difficulty and Realism

This competition includes meaningful challenges:

- Class imbalance: positive class rate is ~12.44% on train+val.
- Graph sparsity: the graph is very sparse (density ~0.013%).
- Missingness indicators: features include missingness flags (e.g., `HbA1c_missing`).

---

## Evaluation Metric

Macro F1-score with threshold = 0.5

- Evaluates both classes equally
- Strongly penalizes predicting only the majority class

Leaderboard ranking follows competition-style rules: tied scores share the same rank (1,2,2,4).

---

## Repository Data

### Public data (committed)

```text
data/public/
├── nodes.csv           # node features ONLY (no labels)
├── edges.csv           # adjacency list (graph edges)
├── train.csv           # node ids + labels (training)
├── val.csv             # node ids + labels (validation)
├── test_nodes.csv      # node ids ONLY (final test for predictions)
├── sample_submission.csv
```

### Private data (never committed)

```text
data/private/
└── test_labels.csv     # hidden ground truth (used only in CI scoring)
```

Test labels are injected securely during GitHub Actions via Secrets.

---

## Submission (STRICT)

### Required folder structure

```text
submissions/inbox/<team>/<run_id>/
├── predictions.csv
└── metadata.json
```

### predictions.csv format

```csv
id,y_pred
123,0.82
124,0.11
125,0.93
```

Rules:

- `id` must match exactly the ids in `data/public/test_nodes.csv`
- `y_pred` must be a probability in [0, 1]
- Row count must match `test_nodes.csv`

### metadata.json format

Copy from:

```text
submissions/inbox/metadata_template.json
```

Example:

```text
{
  "team": "my_team",
  "run_id": "gcn_v1",
  "author_type": "human",
  "model": "GCN + MLP",
  "notes": "Used edges + class weights"
}
```

Allowed author_type values:

```text
human
llm
hybrid
```

---

## Submission Policy

- One submission attempt per participant (enforced in CI).
- Do not submit code - predictions only.
- PRs must modify only the two submission files.

---

## LLM Usage Restriction

Large Language Models must not be used to fully design the competition, including dataset creation, task definition, or evaluation logic.

---

## Computational Affordability

Full training should not exceed 3 hours on CPU for this competition. If your approach requires more, it is not permitted.

---

## Duplicate Competitions

Overlapping or duplicate competitions will be merged, or only the best-designed version will be retained.

---

## Automatic Scoring (GitHub Actions)

On Pull Request:

- Validates folder structure
- Validates metadata.json
- Scores predictions.csv
- Posts Macro-F1 + metadata as a PR comment
- Does not update leaderboard yet

On merge to main:

- Detects the merged submission
- Scores it again
- Updates and commits the leaderboard

---

## Leaderboard

Source of truth: GitHub Pages at `docs/leaderboard.html` (data from `docs/leaderboard.csv`).

Markdown: `leaderboard/leaderboard.md`  
Interactive UI: `docs/leaderboard.html`

---

## Rules (Summary)

- No external data
- Do not modify:
  - `data/`
  - `scoring/`
  - `.github/workflows/`
  - `leaderboard/`
- Train locally
- Submit only predictions.csv + metadata.json in the correct folder

Invalid submissions will be rejected automatically.
