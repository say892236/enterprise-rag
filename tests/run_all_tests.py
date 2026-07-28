# tests/run_all_tests.py

"""
统一测试入口。

作用：
1. 依次运行当前项目的核心测试。
2. 任意一个测试失败，就立即停止。
3. 方便每次重要修改后做完整检查。

运行方式：
python -m tests.run_all_tests
"""

import subprocess
import sys
from typing import List


# 需要依次执行的测试模块
TEST_MODULES: List[str] = [
    "tests.test_auth",
    "tests.test_access_control",
    "tests.test_confidence_guard",
    "tests.test_milvus_store",
    "tests.test_milvus_retriever",
    "tests.smoke_test",

]


def run_test_module(module_name: str) -> None:
    """
    运行单个测试模块。

    使用 subprocess 的原因：
    - 每个测试模块独立运行
    - 输出更接近手动执行 python -m xxx
    - 避免不同测试之间共享内存状态
    """

    print("=" * 100)
    print(f"开始运行测试：{module_name}")
    print("=" * 100)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
        ],
        text=True,
    )

    if result.returncode != 0:
        print("=" * 100)
        print(f"测试失败：{module_name}")
        print("=" * 100)

        # 直接退出，并返回失败状态码
        sys.exit(result.returncode)

    print("=" * 100)
    print(f"测试通过：{module_name}")
    print("=" * 100)
    print()


def main():
    """
    依次运行所有测试。
    """

    print("=" * 100)
    print("开始执行项目统一测试")
    print("=" * 100)

    for module_name in TEST_MODULES:
        run_test_module(module_name)

    print("=" * 100)
    print("全部测试通过")
    print("=" * 100)


if __name__ == "__main__":
    main()