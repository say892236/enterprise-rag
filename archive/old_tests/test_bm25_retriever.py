# test_bm25_retriever.py

"""
BM25 Retriever 测试脚本。

目标：
1. 确认 chunks.jsonl 能正常加载。
2. 确认 BM25 能返回结果。
3. 测试编号、金额、关键词类查询效果。
"""

# 测试项	结果	判断
# FIN-TR-2026-01	Top1 命中差旅制度文档头	正常
# FAQ-003	命中多个制度里的 FAQ-003	正常，但查询歧义
# 编号类查询	BM25 有明显优势	正常
# 是否需要 metadata filter	需要	企业项目必备

from retriever.bm25_retriever import BM25Retriever


def print_docs(query: str, docs):
    """
    打印 BM25 检索结果。

    这里用 .get() 读取 metadata，
    避免某些 chunk 没有 header_2 时出现 KeyError。
    """

    print("=" * 100)
    print("查询问题：")
    print(query)

    print("\n检索结果：")

    for i, doc in enumerate(docs, start=1):
        print(f"\n【结果 {i}】")
        print("source:", doc.metadata.get("source"))
        print("header_1:", doc.metadata.get("header_1"))
        print("header_2:", doc.metadata.get("header_2"))
        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("bm25_score:", doc.metadata.get("bm25_score"))

        print("\n内容预览：")
        print(doc.page_content[:300])
        print("-" * 100)


def main():
    # 1. 初始化 BM25 检索器
    # 它会从 data/processed/chunks.jsonl 加载全部 chunks
    retriever = BM25Retriever(
        chunks_file="data/processed/chunks.jsonl",
        top_k=5,
    )

    print("BM25 chunks count:", retriever.count())

    # 2. 测试问题
    # 这些问题更偏关键词、编号、金额，正适合测试 BM25
    test_queries = [
        "FAQ-003",
        "FIN-TR-2026-01",
        "单次市内打车超过 200 元",
        "上海 普通员工 600 元",
        "10 个工作日 提交报销",
        "住宿标准",
    ]

    # 3. 逐个问题测试
    for query in test_queries:
        docs = retriever.retrieve(
            query=query,
            top_k=5,
        )

        print_docs(query, docs)

    # 4. 测试 metadata filter
    # 场景：FAQ-003 在多个制度文档里都存在。
    # 如果不加 source 过滤，BM25 不知道用户要哪个制度里的 FAQ-003。
    # 所以这里限定只查 travel_policy.md。
    filtered_docs = retriever.retrieve(
        query="FAQ-003",
        top_k=5,
        metadata_filter={"source": "travel_policy.md"},
    )

    print_docs(
        query="FAQ-003 + source=travel_policy.md",
        docs=filtered_docs,
    )


if __name__ == "__main__":
    main()