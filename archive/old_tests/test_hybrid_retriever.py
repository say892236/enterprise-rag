# test_hybrid_retriever.py

"""
Hybrid Retriever 测试脚本。

目标：
1. 验证 Chroma + BM25 能否同时召回。
2. 验证融合结果是否去重。
3. 观察每个 chunk 是 Dense 召回、BM25 召回，还是两者都召回。
4. 为后续接入 Reranker 做准备。
"""

from retriever.chroma_retriever import ChromaRetriever
from retriever.bm25_retriever import BM25Retriever
from retriever.hybrid_retriever import HybridRetriever


def print_docs(query: str, docs):
    """
    打印 Hybrid 检索结果。

    注意：
    - hybrid_score 越高越相关。
    - chroma_score 越低越相关。
    - bm25_score 越高越相关。
    """

    print("=" * 120)
    print("查询问题：")
    print(query)

    print("\nHybrid 检索结果：")

    for i, doc in enumerate(docs, start=1):
        print(f"\n【结果 {i}】")
        print("source:", doc.metadata.get("source"))
        print("header_1:", doc.metadata.get("header_1"))
        print("header_2:", doc.metadata.get("header_2"))
        print("chunk_id:", doc.metadata.get("chunk_id"))

        print("retrieval_sources:", doc.metadata.get("retrieval_sources"))

        print("hybrid_score:", doc.metadata.get("hybrid_score"))

        print("dense_rank:", doc.metadata.get("dense_rank"))
        print("chroma_score:", doc.metadata.get("chroma_score"))
        print("dense_rrf_score:", doc.metadata.get("dense_rrf_score"))

        print("bm25_rank:", doc.metadata.get("bm25_rank"))
        print("bm25_score:", doc.metadata.get("bm25_score"))
        print("bm25_rrf_score:", doc.metadata.get("bm25_rrf_score"))

        print("\n内容预览：")
        print(doc.page_content[:300])
        print("-" * 120)


def main():
    # 1. 初始化 Chroma Dense Retriever
    # 负责语义检索。
    dense_retriever = ChromaRetriever(
        persist_directory="data/vectorstore/chroma",
        collection_name="enterprise_rag",
        model_name="BAAI/bge-m3",
        device="cpu",
        top_k=10,
    )

    # 2. 初始化 BM25 Retriever
    # 负责关键词、编号、金额类检索。
    bm25_retriever = BM25Retriever(
        chunks_file="data/processed/chunks.jsonl",
        top_k=10,
    )

    # 3. 初始化 Hybrid Retriever
    # 同时调用 Dense 和 BM25，并用 RRF 做排名融合。
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=5,
        dense_weight=1.0,
        bm25_weight=1.0,
        rrf_k=60,
    )

    # 4. 测试问题
    # 包含语义问题、编号问题、金额问题。
    test_queries = [
        "员工出差住宿标准是什么？",
        "住宿能报多少？",
        "FIN-TR-2026-01",
        "FAQ-003",
        "单次市内打车超过 200 元",
        "10 个工作日 提交报销",
    ]

    # 5. 执行测试
    for query in test_queries:
        docs = hybrid_retriever.retrieve(
            query=query,
            top_k=5,
        )

        print_docs(
            query=query,
            docs=docs,
        )


if __name__ == "__main__":
    main()