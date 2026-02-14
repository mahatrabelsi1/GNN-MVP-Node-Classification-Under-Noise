# DiaGraph - Robust Diabetes Prediction on Noisy Graphs

DiaGraph is a prediction-only graph ML mini-competition.

Participants train locally and submit encrypted predictions via Pull Request. GitHub Actions decrypts, scores, and updates the leaderboard automatically.

## Core Task

- Problem: supervised node classification
- Target: `diabetes` (binary)
- Public inputs: node features + graph edges
- Hidden target: test labels used only in CI
- Metric: Macro F1 at threshold 0.5

## Graph Specification

- Node feature matrix `X`: `data/public/nodes.csv`
- Adjacency matrix `A`: `data/public/edges.csv` with (`src`, `dst`)
- Node ids are in column `id` and must align across files

## Data Layout

```text
data/public/
  nodes.csv
  edges.csv
  train.csv
  val.csv
  test_nodes.csv
  sample_submission.csv
```

Private data is never committed:

```text
data/private/test_labels.csv
```

## Security Model

Private submissions must not be publicly readable.

- Participants submit only encrypted prediction files (`predictions.csv.enc`)
- Public key is published in `encryption/public_key.pem`
- Private key stays in GitHub Secrets only
- CI decrypts in runner memory/disk, scores, updates leaderboard, and removes decrypted artifacts

## Submission Format

Run folder:

```text
submissions/inbox/<team>/<run_id>/
  predictions.csv.enc
  metadata.json
```

`metadata.json` required fields:

- `team`
- `run_id`
- `author_type` in `human|llm|hybrid`
- `model`
- `notes` (optional)

Plain predictions schema before encryption:

```csv
id,y_pred
123,0.82
124,0.11
125,0.93
```

Rules:

- `id` must match `data/public/test_nodes.csv` exactly
- `y_pred` must be in `[0, 1]`
- Exactly one attempt per team (enforced in CI)

## How Participants Submit

1. Generate plain predictions locally (`predictions.csv` with `id,y_pred`)
2. Encrypt with repository public key:

```bash
python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
```

3. Place `predictions.csv.enc` and `metadata.json` under:

```text
submissions/inbox/<team>/<run_id>/
```

4. Open PR to `main` with only those 2 files

What CI does on PR:

- validates changed paths and metadata
- restores hidden labels and private key from secrets
- decrypts `predictions.csv.enc`
- computes Macro-F1
- updates `leaderboard/leaderboard.csv`, `leaderboard/leaderboard.md`, `docs/leaderboard.csv` on `main`
- comments score on PR and closes PR (no manual merge needed)

## Maintainer Key Setup

Generate key pair once:

```bash
python encryption/generate_keys.py
```

- Commit only `encryption/public_key.pem`
- Do not commit `encryption/private_key.pem`

Store private key in repo secrets as base64 chunks:

- `SUBMISSION_PRIVATE_KEY_B64_01` ... `SUBMISSION_PRIVATE_KEY_B64_08`

Also keep hidden test labels secrets:

- `TEST_LABELS_B64_01` ... `TEST_LABELS_B64_08`

PowerShell helper to create chunk values from the private key:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("encryption/private_key.pem"))
$size = [Math]::Ceiling($b64.Length / 8)
0..7 | ForEach-Object { $b64.Substring($_ * $size, [Math]::Min($size, $b64.Length - ($_ * $size))) }
```

## Manual Rebuild

Manual workflow `Rebuild Leaderboard (manual)` decrypts all encrypted submissions and rebuilds leaderboard from scratch.

## Policy Notes

- No external data
- LLMs must not fully design the competition dataset/task/evaluation logic
- Intended CPU affordability target: full training <= 3 hours
- Tied scores share rank (competition ranking style)
