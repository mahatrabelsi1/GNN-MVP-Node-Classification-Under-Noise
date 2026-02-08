# scoring/update_leaderboard_all.py
import os
import glob
import subprocess
import sys

INBOX_ROOT = "submissions/inbox"

def main():
    pred_paths = glob.glob(os.path.join(INBOX_ROOT, "*", "*", "predictions.csv"))
    if not pred_paths:
        print("No submissions found under submissions/inbox/<team>/<run_id>/predictions.csv")
        return

    # score/update each run (team/run_id) one by one
    for pred in sorted(pred_paths):
        print("Updating leaderboard for:", pred)
        subprocess.check_call([sys.executable, "scoring/update_leaderboard.py", "--pred", pred])

if __name__ == "__main__":
    main()
