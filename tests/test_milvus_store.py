# tests/test_milvus_store.py

"""
Milvus Store 封装测试。

作用：
1. 测试 Milvus Store 模块能正常导入。
2. 测试 Milvus connection_args 配置正确。
3. 不构建真实向量库，不写入 Milvus 数据。
"""

import sys

from config.settings import (
    MILVUS_COLLECTION_NAME,
    MILVUS_URI,
)
from vectorstore.milvus_store import (
    get_milvus_connection_args,
)


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def test_milvus_config_not_empty() -> None:
    """
    测试 Milvus 基础配置不为空。
    """

    assert MILVUS_URI
    assert MILVUS_COLLECTION_NAME

    print_pass("Milvus 基础配置不为空")


def test_milvus_connection_args() -> None:
    """
    测试 Milvus connection_args 正确。
    """

    connection_args = get_milvus_connection_args()

    assert isinstance(connection_args, dict)
    assert connection_args["uri"] == MILVUS_URI

    print_pass("Milvus connection_args 正确")


def run_all_tests() -> None:
    """
    运行 Milvus Store 测试。
    """

    print("=" * 100)
    print("开始运行 Milvus Store 测试")
    print("=" * 100)

    test_milvus_config_not_empty()
    test_milvus_connection_args()

    print("=" * 100)
    print("Milvus Store 测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)