# tests/test_milvus_retriever.py

"""
Milvus Retriever 单元测试。

作用：
1. 测试 metadata_filter 到 Milvus expr 的转换逻辑。
2. 不连接 Milvus。
3. 不执行真实向量检索。
"""

import sys

from retriever.milvus_retriever import MilvusRetriever


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def test_build_source_expr_none() -> None:
    """
    测试没有 metadata_filter 时，不生成 expr。
    """

    expr = MilvusRetriever._build_source_expr(
        self=None,
        metadata_filter=None,
    )

    assert expr is None

    print_pass("metadata_filter=None 时 expr=None")


def test_build_source_expr_exact_source() -> None:
    """
    测试 source 精确匹配。
    """

    expr = MilvusRetriever._build_source_expr(
        self=None,
        metadata_filter={
            "source": "差旅住宿与交通标准.md"
        },
    )

    assert expr == 'source == "差旅住宿与交通标准.md"'

    print_pass(
        "source 精确匹配 expr 正确"
    )

def test_build_source_expr_source_in() -> None:

    expr = MilvusRetriever._build_source_expr(
        self=None,
        metadata_filter={
            "source": {
                "$in": [
                    "员工手册.md",
                    "差旅住宿与交通标准.md",
                ]
            }
        },
    )

    assert expr == (
        'source in '
        '["员工手册.md", "差旅住宿与交通标准.md"]'
    )

    print_pass(
        "source.$in expr 正确"
    )

def test_build_source_expr_unsupported_filter():

    expr = MilvusRetriever._build_source_expr(
        self=None,
        metadata_filter={
            "unknown_field": "xxx"
        },
    )

    assert expr is None

    print_pass(
        "暂不支持字段返回 expr=None"
    )

def run_all_tests() -> None:
    """
    运行 Milvus Retriever 单元测试。
    """

    print("=" * 100)
    print("开始运行 Milvus Retriever 单元测试")
    print("=" * 100)

    test_build_source_expr_none()
    test_build_source_expr_exact_source()
    test_build_source_expr_source_in()
    test_build_source_expr_unsupported_filter()

    print("=" * 100)
    print("Milvus Retriever 单元测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)