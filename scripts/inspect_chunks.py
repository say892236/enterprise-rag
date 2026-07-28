import json

with open(
"data/processed/chunks.jsonl",
encoding="utf-8"
) as f:

    item=json.loads(next(f))

print(item["metadata"])