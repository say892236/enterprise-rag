from pymilvus import connections, Collection


connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)


collection = Collection(
    "enterprise_rag"
)

collection.load()


# 不使用 num_entities
results = collection.query(
    expr="",
    output_fields=[
        "text",
        "source",
        "department",
        "document_id"
    ],
    limit=3
)


for item in results:
    print("="*50)
    print(item)

print(
    "真实实体数量:",
    len(results)
)


print("="*50)


print(results[0])