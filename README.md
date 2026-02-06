# 🧠 DiaGraph — Robust Diabetes Prediction on Noisy Graphs

**DiaGraph** is a **prediction-only GitHub mini-competition** for **graph machine learning**.

It is **not an application**.  
It is a **benchmark + evaluation pipeline** where participants train models **locally** and submit **only prediction files** via Pull Requests.

---

## ✅ What this project is (exactly)

This project is:

- **Supervised Learning** (train/val labels are provided)
- **Graph Machine Learning** task
- **Node Classification** problem  
  (each node = one patient, predict diabetes label per node)
- A **robustness / noise-oriented benchmark**  
  (noisy + imbalanced data, hidden test labels)

**One-line description:**

> **A supervised graph node classification benchmark for robust diabetes prediction under noisy conditions.**

---

## 🎯 Objective

Predict whether a node represents a **diabetic patient** (binary classification) using:

- **Node features** (clinical attributes)
- **Graph edges / adjacency** (similarity relations between patients)

Both **GNN models** and **non-GNN baselines** are allowed.

---

## 🧩 Graph Setting

- Each row in `nodes.csv` represents one **patient node**
- `edges.csv` contains the **graph structure**
- Your model can use edges (GNNs) or ignore them (baseline)

This is a **graph node classification** task.

---

## 📂 Repository Data

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

That’s the right Markdown structure: **one fenced block** only.


### 🔒 Private data (never committed)
```text
data/private/
└── test_labels.csv # hidden ground truth (used only in CI scoring)
```
Test labels are injected securely during GitHub Actions via Secrets.

---

## 📊 Evaluation Metric

**Macro F1-score** with **threshold = 0.5**

- evaluates both classes equally
- strongly penalizes predicting only the majority class

---

## 📤 Submission (STRICT)

### Required folder structure
```text
submissions/inbox/<team>/<run_id>/
├── predictions.csv
└── metadata.json
```

### `predictions.csv` format

```csv
id,y_pred
123,0.82
124,0.11
125,0.93
```
**Rules**

`id` must match exactly the ids in `data/public/test_nodes.csv`  
`y_pred` must be a probability in \[0, 1]  
Row count must match `test_nodes.csv`  

---

**metadata.json format**

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

### 🚀 How to Participate
1/ Clone the repository
2/Train your model locally using:
```text
data/public/nodes.csv

data/public/edges.csv

data/public/train.csv + val.csv
```
3/Generate predictions for all ids in data/public/test_nodes.csv
Create:
```text
predictions.csv

metadata.json
```
Add them to:
```text
submissions/inbox/<team>/<run_id>/
```
4/Open a Pull Request
✅ the PR must modify only these two files


### 🤖 Automatic Scoring (GitHub Actions)
On Pull Request
```text
validates folder structure

validates metadata.json

scores predictions.csv

posts Macro-F1 + metadata as a PR comment

❌ does not update leaderboard yet

On Merge to main
detects the merged submission

scores it again

updates and commits
```

### 🏆 Leaderboard
Source of truth: leaderboard/leaderboard.csv

Markdown: leaderboard/leaderboard.md

Interactive UI:
docs/leaderboard.html


### 📜 Rules
```text
❌ No external data
❌ Do not modify:
data/
scoring/
.github/workflows/
leaderboard/

❌ Do not submit code — predictions only
❌ One submission per Pull Request

✅ Train locally
✅ Submit only predictions.csv + metadata.json in the correct folder

Invalid submissions will be rejected automatically.
```
