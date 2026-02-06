# Submissions (Inbox)

Each submission must be added via Pull Request in this exact structure:

submissions/inbox/<team_name>/<run_id>/predictions.csv
submissions/inbox/<team_name>/<run_id>/metadata.json

## predictions.csv format
Must contain **exactly** these columns:

id,y_pred

- `id` must match `data/public/test_nodes.csv` exactly (same IDs)
- `y_pred` must be a number in [0, 1]

## metadata.json format
Example:

{
  "team": "demo_team",
  "run_id": "run1",
  "author_type": "human",
  "model": "GCN",
  "notes": "optional short note"
}

Allowed values for `author_type`:
- "human"
- "llm"
- "hybrid"
