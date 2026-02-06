import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

TEST_NODES = "data/public/test_nodes.csv"
TEST_LABELS = "data/private/test_labels.csv"  # local only (never commit)
INBOX_ROOT = "submissions/inbox"

THRESHOLD = 0.5


def find_latest_predictions():
    pattern = os.path.join(INBOX_ROOT, "*", "*", "predictions.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            "No submissions found. Expected: submissions/inbox/<team>/<run_id>/predictions.csv"
        )
    return max(paths, key=os.path.getmtime)


def score_file(pred_path: str) -> float:
    test_ids = pd.read_csv(TEST_NODES)
    y_true_df = pd.read_csv(TEST_LABELS)
    pred = pd.read_csv(pred_path)

    # validate columns
    if list(pred.columns) != ["id", "y_pred"]:
        raise ValueError("predictions.csv must have exactly columns: id,y_pred")

    # validate row count
    if len(pred) != len(test_ids):
        raise ValueError(f"Wrong number of rows. Expected {len(test_ids)}, got {len(pred)}")

    # validate ids
    if set(pred["id"]) != set(test_ids["id"]):
        raise ValueError("IDs in predictions.csv must match data/public/test_nodes.csv exactly")

    # align order
    pred = pred.set_index("id").loc[test_ids["id"]].reset_index()
    y_true = y_true_df.set_index("id").loc[test_ids["id"]]["diabetes"].astype(int).values

    y_prob = pred["y_pred"].astype(float).values
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_pred must be in [0, 1]")

    y_hat = (y_prob >= THRESHOLD).astype(int)
    return float(f1_score(y_true, y_hat, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default="", help="Path to predictions.csv (optional)")
    args = ap.parse_args()

    pred_path = args.pred.strip() if args.pred else ""
    if pred_path:
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"--pred file not found: {pred_path}")
    else:
        pred_path = find_latest_predictions()

    print("📄 Using submission:", pred_path)
    score = score_file(pred_path)
    print("✅ Macro-F1:", round(score, 6))


if __name__ == "__main__":
    main()
