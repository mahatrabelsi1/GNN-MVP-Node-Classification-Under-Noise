import glob
import os
import subprocess
import sys
import tempfile

INBOX_ROOT = "submissions/inbox"
PRIVATE_KEY = "data/private/submission_private_key.pem"


def main() -> None:
    enc_paths = sorted(glob.glob(os.path.join(INBOX_ROOT, "*", "*", "predictions.csv.enc")))
    if not enc_paths:
        print("No encrypted submissions found under submissions/inbox/<team>/<run_id>/predictions.csv.enc")
        return

    for enc_path in enc_paths:
        run_dir = os.path.dirname(enc_path)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata.json next to encrypted predictions: {meta_path}")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_pred = tmp.name

        try:
            subprocess.check_call([sys.executable, "encryption/decrypt.py", enc_path, PRIVATE_KEY, tmp_pred])
            subprocess.check_call(
                [sys.executable, "scoring/update_leaderboard.py", "--pred", tmp_pred, "--meta", meta_path]
            )
        finally:
            if os.path.exists(tmp_pred):
                os.remove(tmp_pred)


if __name__ == "__main__":
    main()
