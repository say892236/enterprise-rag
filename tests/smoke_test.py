# tests/smoke_test.py

"""
项目核心链路 Smoke Test。

Smoke Test 的目标不是全面评估效果，而是快速确认：
1. Chroma 向量库能正常加载。
2. Dense Retriever 能正常返回结果。
3. BM25 Retriever 能正常返回结果。
4. Hybrid Retriever 能正常融合 Dense + BM25 结果。
5. 当前核心模块没有因为重构而坏掉。

运行方式：
python -m tests.smoke_test

注意：
这里默认不调用 LLM，也不跑 Reranker。
原因：
- LLM 需要 API，速度慢。
- Reranker CPU 推理慢。
- Smoke Test 应该尽量轻量、快速。
"""

from pipelines.rag_pipeline import (
    build_dense_retriever,
    build_bm25_retriever,
    build_hybrid_retriever,
)


def assert_not_empty(name: str, docs: list):
    """
    检查检索结果不能为空。

    参数：
    name:
        当前测试项名称，用于报错提示。

    docs:
        检索返回的 Document 列表。
    """

    assert docs, f"{name} 返回结果为空"


def test_dense_retriever():
    """
    测试 Dense Retriever。

    目标：
    确认 Chroma 向量库能加载，并且能根据语义问题召回文档。
    """

    retriever = build_dense_retriever()

    docs = retriever.retrieve(
        query="员工出差住宿标准是什么？",
        top_k=5,
    )

    assert_not_empty("Dense Retriever", docs)

    print("[PASS] Dense Retriever 正常")
    print("Top1:", docs[0].metadata.get("source"), docs[0].metadata.get("header_2"))


def test_bm25_retriever():
    """
    测试 BM25 Retriever。

    目标：
    确认 BM25 对制度编号这类精确关键词查询有效。
    """

    retriever = build_bm25_retriever()

    docs = retriever.retrieve(
        query="FIN-TR-2026-01",
        top_k=5,
    )

    assert_not_empty("BM25 Retriever", docs)

    top1 = docs[0]

    # FIN-TR-2026-01 应该命中差旅制度文档头
    assert top1.metadata.get("source") == "travel_policy.md", (
        "BM25 编号检索 Top1 不符合预期，"
        f"实际为: {top1.metadata.get('source')}"
    )

    print("[PASS] BM25 Retriever 正常")
    print("Top1:", top1.metadata.get("source"), top1.metadata.get("chunk_id"))


def test_hybrid_retriever():
    """
    测试 Hybrid Retriever。

    目标：
    确认 Dense + BM25 可以一起工作，并且通过 RRF 正常融合。
    """

    retriever = build_hybrid_retriever()

    docs = retriever.retrieve(
        query="FIN-TR-2026-01",
        top_k=5,
    )

    assert_not_empty("Hybrid Retriever", docs)

    top1 = docs[0]

    # Hybrid 查询制度编号时，Top1 也应该是差旅制度文档头
    assert top1.metadata.get("source") == "travel_policy.md", (
        "Hybrid 编号检索 Top1 不符合预期，"
        f"实际为: {top1.metadata.get('source')}"
    )

    retrieval_sources = top1.metadata.get("retrieval_sources", [])

    # 这个编号查询理想情况下 Dense 和 BM25 都会命中
    assert "bm25" in retrieval_sources, (
        "Hybrid 结果中没有 BM25 来源，"
        f"retrieval_sources={retrieval_sources}"
    )

    print("[PASS] Hybrid Retriever 正常")
    print("Top1:", top1.metadata.get("source"), top1.metadata.get("chunk_id"))
    print("retrieval_sources:", retrieval_sources)


def main():
    """
    统一执行所有 Smoke Test。
    """

    print("=" * 100)
    print("开始执行 Smoke Test")
    print("=" * 100)

    test_dense_retriever()
    test_bm25_retriever()
    test_hybrid_retriever()

    print("=" * 100)
    print("全部 Smoke Test 通过")
    print("=" * 100)


if __name__ == "__main__":
    main()