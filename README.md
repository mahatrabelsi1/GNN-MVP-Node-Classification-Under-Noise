# DiaGraph - Robust Diabetes Prediction on Noisy Graphs

**DiaGraph** is a **prediction-only GitHub mini-competition** for **graph machine learning**.

It is **not an application**.  
It is a **benchmark + evaluation pipeline** where participants train models **locally** and submit **only encrypted prediction files** via Pull Requests.

---

## What this project is (exactly)

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

## Objective

Predict whether a node represents a **diabetic patient** (binary classification) using:

- **Node features** (clinical attributes)
- **Graph edges / adjacency** (similarity relations between patients)

Both **GNN models** and **non-GNN baselines** are allowed.

---
## Leaderboard

Live leaderboard:  
<https://mahatrabelsi1.github.io/GNN-MVP-Node-Classification-Under-Noise/leaderboard.html>

Source files:

- `leaderboard/leaderboard.csv`
- `leaderboard/leaderboard.md`
- `docs/leaderboard.csv`
- `docs/leaderboard.html`

---

## Graph Setting

- Each row in `nodes.csv` represents one **patient node**
- `edges.csv` contains the **graph structure**
- Your model can use edges (GNNs) or ignore them (baseline)

This is a **graph node classification** task.

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

## Evaluation Metric

**Macro F1-score** with **threshold = 0.5**

- evaluates both classes equally
- strongly penalizes predicting only the majority class

Leaderboard ranking uses competition-style ties (same score => same rank).

---

## Security and Privacy

Private submissions are submitted in encrypted form:

- Participants encrypt predictions locally with public key `encryption/public_key.pem`
- Repo stores only `predictions.csv.enc` (encrypted payload)
- GitHub Actions restores private decryption key from Secrets
- Decryption and scoring happen only inside runner environment
- Leaderboard updates are committed to `main`
- Submission PR is commented and auto-closed (no manual merge needed)

---

## Submission (STRICT)

### Required folder structure

```text
submissions/inbox/<team>/<run_id>/
├── predictions.csv.enc
└── metadata.json
```

### Plain `predictions.csv` format (before encryption)

```csv
id,y_pred
123,0.82
124,0.11
125,0.93
```

### Encryption command

```bash
python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
```

**Rules**

- `id` must match exactly the ids in `data/public/test_nodes.csv`
- `y_pred` must be a probability in [0, 1]
- Row count must match `test_nodes.csv`
- Submit encrypted file only (`predictions.csv.enc`), not plain `predictions.csv`

---

### `metadata.json` format

Copy from:

```text
submissions/inbox/metadata_template.json
```

Example:

```json
{
  "team": "my_team",
  "run_id": "gcn_v1",
  "author_type": "human",
  "model": "GCN + MLP",
  "notes": "Used edges + class weights"
}
```

Allowed `author_type` values:

```text
human
llm
hybrid
```

---

## How to Participate

1. Clone the repository.
2. Train your model locally using:
```text
data/public/nodes.csv
data/public/edges.csv
data/public/train.csv + val.csv
```
3. Generate predictions for all ids in `data/public/test_nodes.csv`.
4. Encrypt predictions:
```bash
python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
```
5. Create:
```text
predictions.csv.enc
metadata.json
```
6. Add them to:
```text
submissions/inbox/<team>/<run_id>/
```
7. Open a Pull Request to `main`.

The PR must modify only these two files.

---

## Automatic Scoring (GitHub Actions)

On Pull Request:

- validates folder structure
- validates `metadata.json`
- restores hidden labels and decryption key from Secrets
- decrypts encrypted submission
- scores Macro-F1
- updates leaderboard files directly on `main`
- posts Macro-F1 + metadata as a PR comment
- closes PR automatically

No manual merge is required for submission PRs.

---



## Rules

- No external data
- Do not submit training code; predictions + metadata only
- One submission attempt per team (enforced in CI)
- LLMs must not fully design dataset/task/evaluation logic

Invalid submissions are rejected automatically.
