# 🧠 GNN-MVP: Node Classification Under Noise

Welcome to the **GNN-MVP Node Classification Challenge**.  
This is a lightweight, GitHub-hosted machine learning competition inspired by Kaggle, designed to be **easy to join but hard to win**.

---

## 🎯 Challenge Objective

Your task is to **predict the class label (`target`) for each node** in a test dataset using node-level features.

The dataset is intentionally:
- Noisy
- Imbalanced
- Deceptively simple

Simple models may work — but optimizing the evaluation metric is challenging.

---

## 📂 Dataset Description

All data is provided as CSV files.

### `data/train.csv`
Contains node features and labels.

id, f1, f2, f3, ..., target

shell
Copy code

### `data/test.csv`
Contains node features only.

id, f1, f2, f3, ...

yaml
Copy code

Each row represents **one node**.

---

## 📊 Evaluation Metric

### Macro F1-score

- F1-score is computed independently for each class
- All classes are weighted equally
- Strongly penalizes models that ignore minority classes

➡️ Predicting only the majority class will result in a low score.

---

## 📤 Submission Format

Participants must submit a CSV file in the `submissions/` directory with the **exact format**:

id,target
0,2
1,0
2,1
...

markdown
Copy code

### Submission Rules
- `id` must exactly match the IDs in `test.csv`
- `target` must be a valid class label
- **One submission per Pull Request**

---

## 📤 How to Submit

1. Train your model locally using the provided dataset.
2. Generate predictions for all samples in `data/test.csv`.
3. Create a CSV file with the required format:
id,target

yaml
Copy code
4. Save your file in the `submissions/` directory (example: `submissions/yourname.csv`).
5. Open a Pull Request with **one submission file only**.

Your submission will be scored automatically.

---

## 🚀 Getting Started (Baseline)

A simple baseline model is provided to help you get started and verify the submission format.

### Run the baseline

```bash
cd starter_code
pip install -r requirements.txt
python baseline.py 
```
This script will:

Train a basic model on data/train.csv

Print a validation Macro F1-score

Generate a valid submission file at:
submissions/sample_submission.csv

## 🏆 Leaderboard
Submissions are scored automatically using GitHub Actions.

After each valid Pull Request:

Your submission is evaluated using the Macro F1-score

Results are added to leaderboard.md

Rankings are sorted from highest to lowest score

Only the best score per participant is kept on the leaderboard.
--
## 📜 Rules
To ensure a fair competition, the following rules apply:

❌ No external data is allowed

❌ Do not modify files in the data/ directory

❌ Do not modify scoring or leaderboard scripts

❌ Do not submit code — predictions only

✅ Models must be trained locally

⚠️ Invalid or malformed submissions will be rejected automatically

Failure to follow these rules may result in disqualification.
