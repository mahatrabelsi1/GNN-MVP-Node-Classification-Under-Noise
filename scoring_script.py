import argparse
import os
import sys
import pandas as pd
from sklearn.metrics import f1_score


REQUIRED_COLS = {"id", "target"}


def die(msg: str, code: int = 2):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(code)


def load_csv(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        die(f"{name} file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        die(f"Failed to read {name} CSV: {path}\nError: {e}")
    return df


def validate_columns(df: pd.DataFrame, name: str):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        die(f"{name} missing required columns: {sorted(missing)}. Found: {list(df.columns)}")


def validate_no_duplicates(df: pd.DataFrame, name: str):
    if df["id"].duplicated().any():
        dup_ids = df.loc[df["id"].duplicated(), "id"].head(10).tolist()
        die(f"{name} has duplicated ids. Examples: {dup_ids}")


def main():
    parser = argparse.ArgumentParser(
        description="Score a submission for GNN-MVP using Macro F1."
    )
    parser.add_argument(
        "submission_file",
        help="Path to participant submission CSV (must contain columns: id,target)",
    )
    parser.add_argument(
        "--truth",
        default="data/test_labels.csv",
        help="Path to hidden ground truth CSV (default: data/test_labels.csv)",
    )
    args = parser.parse_args()

    submission = load_csv(args.submission_file, "Submission")
    truth = load_csv(args.truth, "Ground truth")

    validate_columns(submission, "Submission")
    validate_columns(truth, "Ground truth")

    submission = submission[["id", "target"]].copy()
    truth = truth[["id", "target"]].copy()

    validate_no_duplicates(submission, "Submission")
    validate_no_duplicates(truth, "Ground truth")

    # Ensure IDs match exactly (no missing, no extra)
    truth_ids = set(truth["id"].tolist())
    sub_ids = set(submission["id"].tolist())

    missing_ids = truth_ids - sub_ids
    extra_ids = sub_ids - truth_ids

    if missing_ids:
        sample = list(missing_ids)[:10]
        die(f"Submission is missing predictions for some test ids. Examples: {sample}")

    if extra_ids:
        sample = list(extra_ids)[:10]
        die(f"Submission contains unknown ids not in the test set. Examples: {sample}")

    # Merge by id (robust to ordering)
    merged = truth.merge(submission, on="id", how="left", suffixes=("_true", "_pred"))

    if merged["target_pred"].isna().any():
        die("Submission contains empty/NaN predictions in 'target'.")

    # Compute Macro F1
    y_true = merged["target_true"]
    y_pred = merged["target_pred"]

    # Try to coerce numeric labels if user submitted strings like "1"
    try:
        y_pred = pd.to_numeric(y_pred)
    except Exception:
        pass

    score = f1_score(y_true, y_pred, average="macro")

    print(f"✅ Submission Macro F1: {score:.6f}")


if __name__ == "__main__":
    main()
