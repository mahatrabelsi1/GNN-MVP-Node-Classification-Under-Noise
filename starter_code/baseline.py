import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
SUB_DIR = os.path.join(ROOT, "submissions")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUT_PATH = os.path.join(SUB_DIR, "sample_submission.csv")

os.makedirs(SUB_DIR, exist_ok=True)

# Load training data
train = pd.read_csv(TRAIN_PATH)
X = train.drop(columns=["target"])
y = train["target"]

# Drop ID from features
X = X.drop(columns=["id"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Validation
val_pred = model.predict(X_val)
score = f1_score(y_val, val_pred, average="macro")
print(f"Validation Macro F1: {score:.4f}")

# Predict test set
test = pd.read_csv(TEST_PATH)
test_ids = test["id"]
test_features = test.drop(columns=["id"])

test_pred = model.predict(test_features)

submission = pd.DataFrame({
    "id": test_ids,
    "target": test_pred
})

submission.to_csv(OUT_PATH, index=False)
print(f"Submission saved to {OUT_PATH}")
