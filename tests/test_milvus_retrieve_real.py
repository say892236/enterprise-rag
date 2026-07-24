# tests/test_milvus_retrieve_real.py

"""
Milvus 真实检索测试。

运行前要求：
1. .env 中 VECTOR_DB=milvus
2. 已执行 python build_database.py 构建 Milvus 向量库

运行方式：
python -m tests.test_milvus_retrieve_real
"""

import sys

from retriever.milvus_retriever import MilvusRetriever


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def test_milvus_retrieve_travel_policy() -> None:
    """
    测试 Milvus 能检索到差旅制度相关文档。
    """

    retriever = MilvusRetriever()

    docs = retriever.retrieve(
        query="员工出差是否可以入住五星级酒店？",
        top_k=3,
        metadata_filter={
            "source": "travel_policy.md",
        },
    )

    assert len(docs) > 0, "Milvus 没有检索到任何文档"

    returned_sources = {
        doc.metadata.get("source")
        for doc in docs
    }

    assert "travel_policy.md" in returned_sources, (
        f"预期检索到 travel_policy.md，实际 sources={returned_sources}"
    )

    for doc in docs:
        assert "milvus_score" in doc.metadata
        assert "dense_rank" in doc.metadata

    print_pass("Milvus 可以正常检索 travel_policy.md")


def run_all_tests() -> None:
    """
    运行 Milvus 真实检索测试。
    """

    print("=" * 100)
    print("开始运行 Milvus 真实检索测试")
    print("=" * 100)

    test_milvus_retrieve_travel_policy()

    print("=" * 100)
    print("Milvus 真实检索测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)