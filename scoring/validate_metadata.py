import json
import sys

ALLOWED_AUTHOR_TYPES = {"human", "llm", "hybrid"}
REQUIRED_KEYS = {"team", "run_id", "author_type", "model"}

def main():
    if len(sys.argv) != 2:
        print("Usage: python scoring/validate_metadata.py <metadata.json>")
        sys.exit(2)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in data or str(data[k]).strip() == ""]
    if missing:
        print("❌ metadata.json missing required fields:", ", ".join(missing))
        sys.exit(1)

    author_type = str(data.get("author_type", "")).strip().lower()
    if author_type not in ALLOWED_AUTHOR_TYPES:
        print("❌ metadata.json invalid author_type:", author_type)
        print("Allowed:", ", ".join(sorted(ALLOWED_AUTHOR_TYPES)))
        sys.exit(1)

    # Optional consistency checks
    # team/run_id can be anything, but ensure they are strings
    if not isinstance(data["team"], str) or not isinstance(data["run_id"], str):
        print("❌ metadata.json: team and run_id must be strings")
        sys.exit(1)

    print("✅ metadata.json is valid")

if __name__ == "__main__":
    main()
