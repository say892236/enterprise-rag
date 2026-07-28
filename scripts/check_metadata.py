import json


file = "data/processed/chunks.jsonl"


required_fields = [
    "source",
    "department",
    "allowed_roles",
    "document_id",
    "security_level",
    "header_1",
    "chunk_id",
]


total = 0
failed = 0


with open(file,encoding="utf-8") as f:

    for line in f:

        total += 1

        data = json.loads(line)

        metadata = data["metadata"]

        for field in required_fields:

            if field not in metadata:
                print(
                    "缺失:",
                    field,
                    metadata.get("source")
                )

                failed += 1
                break


print("================")
print("总chunk:", total)
print("异常:", failed)