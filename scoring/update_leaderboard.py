# scoring/update_leaderboard.py
import os
import glob
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

TEST_NODES = "data/public/test_nodes.csv"
TEST_LABELS = "data/private/test_labels.csv"   # restored via secret in Actions
INBOX_ROOT = "submissions/inbox"

LB_CSV = "leaderboard/leaderboard.csv"
LB_MD = "leaderboard/leaderboard.md"

THRESHOLD = 0.5


def find_latest_predictions() -> str:
    pattern = os.path.join(INBOX_ROOT, "*", "*", "predictions.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            "No submissions found. Expected: submissions/inbox/<team>/<run_id>/predictions.csv"
        )
    return max(paths, key=os.path.getmtime)


def parse_team_run(pred_path: str) -> tuple[str, str]:
    parts = pred_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return "unknown", "unknown"
    return parts[-3], parts[-2]


def metadata_path_from_pred(pred_path: str) -> str:
    return os.path.join(os.path.dirname(pred_path), "metadata.json")


def load_metadata(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found next to predictions.csv: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required keys (same spirit as validate_metadata.py)
    for k in ["team", "run_id", "author_type", "model"]:
        if k not in data or str(data[k]).strip() == "":
            raise ValueError(f"metadata.json missing required field: {k}")

    author_type = str(data["author_type"]).strip().lower()
    if author_type not in {"human", "llm", "hybrid"}:
        raise ValueError(f"metadata.json invalid author_type: {author_type}")

    # notes is optional
    data["notes"] = str(data.get("notes", "") or "").strip()
    data["author_type"] = author_type
    data["team"] = str(data["team"]).strip()
    data["run_id"] = str(data["run_id"]).strip()
    data["model"] = str(data["model"]).strip()
    return data


def score_submission(pred_path: str) -> float:
    test_ids = pd.read_csv(TEST_NODES)
    y_true_df = pd.read_csv(TEST_LABELS)
    pred = pd.read_csv(pred_path)

    if list(pred.columns) != ["id", "y_pred"]:
        raise ValueError("predictions.csv must have exactly columns: id,y_pred")

    if len(pred) != len(test_ids):
        raise ValueError(f"Wrong number of rows. Expected {len(test_ids)}, got {len(pred)}")

    if set(pred["id"]) != set(test_ids["id"]):
        raise ValueError("IDs in predictions.csv must match data/public/test_nodes.csv exactly")

    pred = pred.set_index("id").loc[test_ids["id"]].reset_index()
    y_true = y_true_df.set_index("id").loc[test_ids["id"]]["diabetes"].astype(int).values

    y_prob = pred["y_pred"].astype(float).values
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_pred must be in [0, 1]")

    y_hat = (y_prob >= THRESHOLD).astype(int)
    return float(f1_score(y_true, y_hat, average="macro"))


def ensure_leaderboard_exists():
    os.makedirs(os.path.dirname(LB_CSV), exist_ok=True)
    if not os.path.exists(LB_CSV):
        cols = ["team", "run_id", "author_type", "model", "notes", "macro_f1", "submitted_at"]
        pd.DataFrame(columns=cols).to_csv(LB_CSV, index=False)


def write_md(lb: pd.DataFrame) -> None:
    lb_sorted = lb.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    lines = []
    lines.append("# Leaderboard\n\n")
    lines.append("| Rank | Team | Run | Author | Model | Macro-F1 | Submitted at |\n")
    lines.append("|---:|---|---|---|---|---:|---|\n")

    last_score = None
    last_rank = 0
    for i, row in lb_sorted.iterrows():
        score = float(row["macro_f1"])
        if i == 0 or score != last_score:
            last_rank = i + 1
            last_score = score
        lines.append(
            f"| {last_rank} | {row['team']} | {row['run_id']} | {row['author_type']} | {row['model']} | "
            f"{score:.6f} | {row['submitted_at']} |\n"
        )

    with open(LB_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default="", help="Path to predictions.csv to score (optional)")
    ap.add_argument("--meta", type=str, default="", help="Path to metadata.json (optional)")
    args = ap.parse_args()

    ensure_leaderboard_exists()

    pred_path = args.pred.strip() if args.pred else ""
    if pred_path:
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"--pred file not found: {pred_path}")
    else:
        pred_path = find_latest_predictions()

    # Read metadata from explicit path or from the same folder as predictions.
    meta_path = args.meta.strip() if args.meta else metadata_path_from_pred(pred_path)
    if args.meta and not os.path.exists(meta_path):
        raise FileNotFoundError(f"--meta file not found: {meta_path}")
    meta = load_metadata(meta_path)

    # Score
    macro_f1 = score_submission(pred_path)

    # Load leaderboard; upgrade old CSV if missing new columns
    lb = pd.read_csv(LB_CSV)

    required_cols = ["team", "run_id", "author_type", "model", "notes", "macro_f1", "submitted_at"]
    for c in required_cols:
        if c not in lb.columns:
            lb[c] = ""

    new_row = {
        "team": meta["team"],
        "run_id": meta["run_id"],
        "author_type": meta["author_type"],
        "model": meta["model"],
        "notes": meta["notes"],
        "macro_f1": macro_f1,
        "submitted_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    # ✅ Deduplicate: keep only the latest for the same (team, run_id)
    # We remove any previous entries with same team+run_id, then append the new one.
    lb = lb[~((lb["team"] == new_row["team"]) & (lb["run_id"] == new_row["run_id"]))].copy()

    lb = pd.concat([lb, pd.DataFrame([new_row])], ignore_index=True)
    lb.to_csv(LB_CSV, index=False)
    write_md(lb)

    print("✅ Leaderboard updated")
    print("Used:", pred_path)
    print("Metadata:", meta_path)
    print("Added/updated:", new_row)


if __name__ == "__main__":
    main()
