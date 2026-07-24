# tests/test_vector_db_config.py

"""
向量数据库配置测试。

作用：
1. 检查 VECTOR_DB 配置是否合法。
2. 检查 Chroma / Milvus 相关配置是否存在。
3. 不连接真实向量库。
"""

import sys

from config.settings import (
    VECTOR_DB,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    MILVUS_URI,
    MILVUS_COLLECTION_NAME,
    MILVUS_DROP_OLD,
)


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def test_vector_db_value() -> None:
    """
    测试 VECTOR_DB 只能是 chroma 或 milvus。
    """

    assert VECTOR_DB in {
        "chroma",
        "milvus",
    }, f"VECTOR_DB 配置非法：{VECTOR_DB}"

    print_pass("VECTOR_DB 配置合法")


def test_chroma_config_exists() -> None:
    """
    测试 Chroma 配置存在。
    """

    assert CHROMA_PERSIST_DIR
    assert CHROMA_COLLECTION_NAME

    print_pass("Chroma 配置存在")


def test_milvus_config_exists() -> None:
    """
    测试 Milvus 配置存在。
    """

    assert MILVUS_URI
    assert MILVUS_COLLECTION_NAME
    assert isinstance(MILVUS_DROP_OLD, bool)

    print_pass("Milvus 配置存在")


def run_all_tests() -> None:
    """
    运行向量数据库配置测试。
    """

    print("=" * 100)
    print("开始运行向量数据库配置测试")
    print("=" * 100)

    test_vector_db_value()
    test_chroma_config_exists()
    test_milvus_config_exists()

    print("=" * 100)
    print("向量数据库配置测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)