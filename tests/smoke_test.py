# tests/smoke_test.py

"""
项目核心链路 Smoke Test。

测试目标：
1. Milvus 向量库可以正常加载。
2. Dense Retriever 可以正常返回语义相关文档。
3. BM25 Retriever 可以根据精确制度编号召回目标文档。
4. Hybrid Retriever 可以融合 Dense 与 BM25 检索结果。
5. 快速确认核心检索链路没有因重构而损坏。

注意：
- Smoke Test 只验证核心链路是否可用，不承担完整的检索效果评估。
- 不调用 LLM，避免产生 API 调用和额外耗时。
- 不调用 Reranker，保持测试轻量。
- 排名质量和 Top1 准确率应由独立评测脚本负责。
"""

from langchain_core.documents import Document

from pipelines.rag_pipeline import (
    build_bm25_retriever,
    build_dense_retriever,
    build_hybrid_retriever,
)


# 当前知识库中已经确认存在的测试文档
TRAVEL_SOURCE = "差旅住宿与交通标准.md"
REIMBURSEMENT_SOURCE = "费用报销制度与流程.md"

# 《费用报销管理制度》中的精确制度编号
REIMBURSEMENT_POLICY_ID = "FIN-EXP-2026-01"


def assert_not_empty(
    name: str,
    docs: list[Document],
) -> None:
    """
    检查检索结果不能为空。
    """

    assert docs, f"{name} 返回结果为空"


def assert_source_in_results(
    name: str,
    docs: list[Document],
    expected_source: str,
) -> None:
    """
    检查目标文档是否出现在检索结果中。

    Smoke Test 重点验证目标文档能够被召回，
    不强制所有语义查询都必须稳定排在 Top1。
    """

    actual_sources = [
        doc.metadata.get("source")
        for doc in docs
    ]

    assert expected_source in actual_sources, (
        f"{name} 未召回目标文档：{expected_source}，"
        f"实际结果：{actual_sources}"
    )


def test_dense_retriever() -> None:
    """
    测试 Dense Retriever。

    使用自然语言问题验证 Milvus Dense 检索
    是否能够召回差旅住宿制度。
    """

    retriever = build_dense_retriever()

    docs = retriever.retrieve(
        query="员工出差住宿标准是什么？",
        top_k=5,
    )

    assert_not_empty(
        name="Dense Retriever",
        docs=docs,
    )

    # 语义查询只要求目标文档进入 Top5，
    # 不在 Smoke Test 中强制要求永远保持 Top1。
    assert_source_in_results(
        name="Dense Retriever",
        docs=docs,
        expected_source=TRAVEL_SOURCE,
    )

    top1 = docs[0]

    print("[PASS] Dense Retriever 正常")
    print(
        "Top1:",
        top1.metadata.get("source"),
        top1.metadata.get("header_2"),
    )


def test_bm25_retriever() -> None:
    """
    测试 BM25 Retriever。

    使用当前知识库真实存在的制度编号，
    验证 BM25 对精确关键词和编号检索有效。
    """

    retriever = build_bm25_retriever()

    docs = retriever.retrieve(
        query=REIMBURSEMENT_POLICY_ID,
        top_k=5,
    )

    assert_not_empty(
        name="BM25 Retriever",
        docs=docs,
    )

    top1 = docs[0]
    top1_source = top1.metadata.get("source")

    # 精确制度编号查询具有较强唯一性，
    # 因此这里可以要求目标文档排在 Top1。
    assert top1_source == REIMBURSEMENT_SOURCE, (
        "BM25 制度编号检索 Top1 不符合预期，"
        f"期望：{REIMBURSEMENT_SOURCE}，"
        f"实际：{top1_source}"
    )

    print("[PASS] BM25 Retriever 正常")
    print(
        "Top1:",
        top1_source,
        top1.metadata.get("chunk_id"),
    )


def test_hybrid_retriever() -> None:
    """
    测试 Hybrid Retriever。

    使用“制度名称 + 精确编号”查询，
    验证 Dense 与 BM25 可以共同参与检索和融合。
    """

    retriever = build_hybrid_retriever()

    docs = retriever.retrieve(
        query=f"费用报销管理制度 {REIMBURSEMENT_POLICY_ID}",
        top_k=5,
    )

    assert_not_empty(
        name="Hybrid Retriever",
        docs=docs,
    )

    # Hybrid Smoke Test 只要求目标文档能够进入 Top5。
    assert_source_in_results(
        name="Hybrid Retriever",
        docs=docs,
        expected_source=REIMBURSEMENT_SOURCE,
    )

    target_doc = next(
        doc
        for doc in docs
        if doc.metadata.get("source") == REIMBURSEMENT_SOURCE
    )

    retrieval_sources = target_doc.metadata.get(
        "retrieval_sources",
        [],
    )

    # 精确编号应该至少由 BM25 检索召回。
    assert "bm25" in retrieval_sources, (
        "Hybrid 目标结果中没有 BM25 检索来源，"
        f"retrieval_sources={retrieval_sources}"
    )

    top1 = docs[0]

    print("[PASS] Hybrid Retriever 正常")
    print(
        "Top1:",
        top1.metadata.get("source"),
        top1.metadata.get("chunk_id"),
    )
    print(
        "目标文档 retrieval_sources:",
        retrieval_sources,
    )


def main() -> None:
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