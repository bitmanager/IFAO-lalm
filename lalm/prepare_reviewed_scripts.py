"""Convert approved dialogue text to the existing MultiTalk TTS input format."""

import argparse
import hashlib
import json
from pathlib import Path


def convert(row, root):
    if row.get("ready_for_audio") is not True:
        raise ValueError(f"Unapproved dialogue: {row['id']}")
    split = row["split"]
    if split not in ("train", "holdout") or Path(row["id"]).name != row["id"]:
        raise ValueError("Invalid split or dialogue ID")
    source = root / split / (row["id"] + ".json")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
        raise ValueError(f"Reviewed source changed: {source}")
    data = json.loads(raw)
    meta = data["meta"]
    if data.get("validation_errors") or meta["language"] != "ru":
        raise ValueError(f"Invalid Russian dialogue: {source}")
    if meta["split"] != split or meta["seed_group"] != row["seed_group"]:
        raise ValueError(f"Source grouping differs from review: {source}")
    names = {"user": "Пользователь", "assistant": "Ассистент"}
    dialogue = []
    for i, turn in enumerate(data["history"]):
        role, text = turn["role"], turn["content"]
        if role != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError(f"Nonalternating dialogue: {source}")
        if not isinstance(text, str) or not text.strip() or "<|" in text:
            raise ValueError(f"Invalid text: {source}")
        dialogue.append({"speaker": names[role], "text": text})
    if len(dialogue) < 4 or len(dialogue) != row["turn_count"]:
        raise ValueError(f"Missing conversational history: {source}")
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "request_id": "reviewed-" + digest[:24],
        "source_group_id": row["seed_group"],
        "split": "validation" if split == "holdout" else "train",
        "config": {"language": "ru"},
        "participants": [
            {"name": name, "role": role,
             "gender": ("male", "female")[int(digest[i], 16) % 2]}
            for i, (role, name) in enumerate(names.items())
        ],
        "dialogue": dialogue,
        "provenance": {"review": row, "scenario_path": str(source)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, groups, ids = [], {}, set()
    for line in args.manifest.read_text().splitlines():
        record = convert(json.loads(line), args.scenario_root)
        group, split = record["source_group_id"], record["split"]
        if groups.setdefault(group, split) != split or record["request_id"] in ids:
            raise ValueError("Duplicate dialogue or source group crossing splits")
        ids.add(record["request_id"])
        records.append(record)
    if not records:
        raise ValueError("No approved dialogues")
    with args.output.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"dialogues": len(records), "source_groups": len(groups)}))


if __name__ == "__main__":
    main()
