# Submissions (Encrypted Inbox)

Each submission PR must add files in this exact structure:

```text
submissions/inbox/<team_name>/<run_id>/predictions.csv.enc
submissions/inbox/<team_name>/<run_id>/metadata.json
```

## Plain predictions schema (before encryption)

```csv
id,y_pred
```

- `id` must match `data/public/test_nodes.csv` exactly
- `y_pred` must be in `[0, 1]`

## Encryption

Encrypt local predictions with repository public key:

```bash
python encryption/encrypt.py predictions.csv encryption/public_key.pem predictions.csv.enc
```

Submit only the encrypted file (`predictions.csv.enc`), not plain CSV.

## metadata.json

Example:

```json
{
  "team": "demo_team",
  "run_id": "run1",
  "author_type": "human",
  "model": "GCN",
  "notes": "optional short note"
}
```

Allowed values for `author_type`:

- `human`
- `llm`
- `hybrid`
