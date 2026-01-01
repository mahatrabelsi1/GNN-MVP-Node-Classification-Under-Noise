import argparse
import datetime as dt
import os
import re
from typing import List, Dict

LEADERBOARD_PATH = "leaderboard.md"

HEADER = """# 🏆 Leaderboard

This leaderboard is updated after each valid submission.

**Metric:** Macro F1-score (higher is better)

| Rank | Participant | Score | Submission | Updated (UTC) |
|------|-------------|-------|------------|---------------|
"""

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*([0-9.]+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")


def read_entries() -> List[Dict]:
    if not os.path.exists(LEADERBOARD_PATH):
        return []

    entries = []
    with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            m = ROW_RE.match(line.strip())
            if m:
                entries.append({
                    "participant": m.group(2).strip(),
                    "score": float(m.group(3)),
                    "submission": m.group(4).strip(),
                    "updated": m.group(5).strip(),
                })
    return entries


def write_entries(entries: List[Dict]):
    # Sort by score desc, tie-break by updated desc
    entries = sorted(entries, key=lambda e: (e["score"], e["updated"]), reverse=True)

    lines = [HEADER]
    for i, e in enumerate(entries, start=1):
        lines.append(
            f"| {i} | {e['participant']} | {e['score']:.6f} | {e['submission']} | {e['updated']} |\n"
        )

    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Update leaderboard.md with a new score entry.")
    parser.add_argument("--participant", required=True, help="Participant name (e.g., GitHub username)")
    parser.add_argument("--score", required=True, type=float, help="Macro F1 score")
    parser.add_argument("--submission", required=True, help="Submission file path (e.g., submissions/user.csv)")
    args = parser.parse_args()

    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    entries = read_entries()

    # Keep best score per participant
    found = False
    for e in entries:
        if e["participant"].lower() == args.participant.lower():
            found = True
            if args.score > e["score"]:
                e["score"] = args.score
                e["submission"] = args.submission
                e["updated"] = now
            break

    if not found:
        entries.append({
            "participant": args.participant,
            "score": args.score,
            "submission": args.submission,
            "updated": now,
        })

    write_entries(entries)
    print("✅ Leaderboard updated.")


if __name__ == "__main__":
    main()
