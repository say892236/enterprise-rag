import json
import random
from collections import Counter


chunk_file = "data/processed/chunks.jsonl"


chunks = []

with open(
    chunk_file,
    "r",
    encoding="utf-8"
) as f:
    for line in f:
        chunks.append(json.loads(line))


print(f"总 chunk 数量: {len(chunks)}")


# =========================
# 1. chunk长度统计
# =========================

lengths = [
    len(chunk["page_content"])
    for chunk in chunks
]


print("\n长度统计:")
print(
    "最小:",
    min(lengths)
)

print(
    "最大:",
    max(lengths)
)

print(
    "平均:",
    sum(lengths) / len(lengths)
)


# =========================
# 2. 查看每个文档产生多少chunk
# =========================

counter = Counter(
    chunk["metadata"]["source"]
    for chunk in chunks
)


print("\n文档chunk数量:")
for name,count in counter.items():
    print(
        name,
        "->",
        count
    )



print("\n随机查看5个chunk")


for chunk in random.sample(chunks,5):

    print("="*80)

    print(
        "source:",
        chunk["metadata"]["source"]
    )

    print(
        "header:",
        chunk["metadata"].get("header_2")
    )

    print(
        "长度:",
        len(chunk["page_content"])
    )


    print(
        chunk["page_content"][:500]
    )