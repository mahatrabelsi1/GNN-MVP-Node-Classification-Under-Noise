import os
import numpy as np
import pandas as pd

# ----------------------------
# CONFIG (locked by our agreement)
# ----------------------------
SEED = 42

N_NODES = 200
N_FEATURES = 20
N_CLASSES = 3
# 
CLASS_PROPORTIONS = [0.50, 0.30, 0.20]  # class 0,1,2
MISSING_RATE = 0.30

TEST_RATIO = 0.35  # % of nodes moved to test set (no labels visible)
NOISE_STD = 0.60   # "good level of noise" (moderate overlap)

OUT_DIR = "data"


def set_seed(seed: int):
    np.random.seed(seed)


def make_labels(n_nodes: int, proportions):
    # Create imbalanced labels with exact counts
    counts = [int(round(p * n_nodes)) for p in proportions]

    # Fix rounding to match exactly n_nodes
    diff = n_nodes - sum(counts)
    counts[0] += diff  # adjust class 0

    y = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    np.random.shuffle(y)
    return y


def generate_features(y: np.ndarray, n_features: int, noise_std: float):
    """
    Create features with:
    - weak individual signals (no single feature separates)
    - stronger combined signal via interactions
    - redundancy + distractor noise features
    """
    n = len(y)
    X = np.random.normal(0, 1.0, size=(n, n_features))

    # Feature groups:
    # 0-5: weak informative (small shifts per class)
    # 6-9: interaction helpers (used in combos)
    # 10-13: redundant (linear combos)
    # 14-19: pure noise distractors

    # Weak informative shifts (small, to avoid single-feature separation)
    shifts = {
        0: np.array([+0.25, -0.10, +0.05, -0.15, +0.10, -0.05]),
        1: np.array([-0.05, +0.20, -0.10, +0.10, -0.10, +0.15]),
        2: np.array([-0.15, -0.10, +0.20, +0.05, +0.10, -0.10]),
    }
    for cls in range(3):
        idx = (y == cls)
        X[idx, 0:6] += shifts[cls]

    # Interaction helpers: create subtle patterns that only help in combination
    # We'll "bend" distributions slightly per class, but still overlapping.
    for cls in range(3):
        idx = (y == cls)
        # slight mean changes + nonlinear transforms
        X[idx, 6] += (cls - 1) * 0.15
        X[idx, 7] += np.tanh(X[idx, 0]) * 0.20
        X[idx, 8] += np.sin(X[idx, 1]) * 0.15
        X[idx, 9] += (X[idx, 2] * X[idx, 3]) * 0.08  # weak product signal

    # Redundant features: linear combos + small noise
    X[:, 10] = 0.7 * X[:, 0] + 0.2 * X[:, 1] + np.random.normal(0, 0.10, n)
    X[:, 11] = -0.6 * X[:, 2] + 0.4 * X[:, 3] + np.random.normal(0, 0.10, n)
    X[:, 12] = 0.5 * X[:, 6] - 0.3 * X[:, 7] + np.random.normal(0, 0.10, n)
    X[:, 13] = 0.4 * X[:, 8] + 0.4 * X[:, 9] + np.random.normal(0, 0.10, n)

    # Pure noise distractors (already random): 14-19 leave as is

    # Add overall moderate noise everywhere (creates overlap)
    X += np.random.normal(0, noise_std, size=X.shape)

    return X


def apply_missingness(X: np.ndarray, missing_rate: float):
    """
    Randomly set ~missing_rate entries to NaN (uniformly).
    """
    X_missing = X.copy()
    mask = np.random.rand(*X_missing.shape) < missing_rate
    X_missing[mask] = np.nan
    return X_missing


def train_test_split_by_id(df: pd.DataFrame, test_ratio: float):
    """
    Shuffle and split by rows, keeping 'id' intact.
    """
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_test = int(round(len(df) * test_ratio))
    test_df = df.iloc[:n_test].copy()
    train_df = df.iloc[n_test:].copy()
    return train_df, test_df


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) IDs
    ids = np.arange(N_NODES)

    # 2) Labels (imbalanced)
    y = make_labels(N_NODES, CLASS_PROPORTIONS)

    # 3) Features with combined-signal difficulty
    X = generate_features(y, N_FEATURES, NOISE_STD)

    # 4) Missing values (30%)
    X = apply_missingness(X, MISSING_RATE)

    # 5) Build dataframe
    feature_cols = [f"f{i+1}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feature_cols)
    df.insert(0, "id", ids)
    df["target"] = y

    # 6) Split into train/test
    train_df, test_df_with_labels = train_test_split_by_id(df, TEST_RATIO)

    # Public files
    train_out = train_df.copy()
    test_out = test_df_with_labels.drop(columns=["target"]).copy()

    # Hidden labels file (organizer-only)
    test_labels_out = test_df_with_labels[["id", "target"]].copy()

    # 7) Save
    train_path = os.path.join(OUT_DIR, "train.csv")
    test_path = os.path.join(OUT_DIR, "test.csv")
    labels_path = os.path.join(OUT_DIR, "test_labels.csv")

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    test_labels_out.to_csv(labels_path, index=False)

    print("✅ Dataset generated successfully:")
    print(f"- {train_path}  (train features + labels)")
    print(f"- {test_path}   (test features only)")
    print(f"- {labels_path} (HIDDEN ground truth; do NOT publish for real competition)")


if __name__ == "__main__":
    main()
