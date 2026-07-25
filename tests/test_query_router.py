# tests/test_query_router.py

"""
Query Router 测试脚本。
"""

import sys
from pathlib import Path

# 把项目根目录加入 Python 模块搜索路径
# 避免误导入 site-packages 里的第三方 retriever 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retriever.query_router import QueryRouter


def main():
    router = QueryRouter()

    questions = [
        "HR-LV-2026-01 是什么制度？",
        "travel_policy.md 里面讲了什么？",
        "普通员工去上海出差，酒店最多能报多少？",
        "员工出差结束后多久内必须提交报销？",
        "出差可以住五星级酒店吗？",
        "公司食堂几点开门？",
        "员工入职需要注意什么？",
    ]

    for question in questions:
        result = router.route(question)

        print("=" * 80)
        print("问题:", question)
        print("查询类型:", result.query_type)
        print("启用 Rewrite:", result.enable_rewrite)
        print("启用 Rerank:", result.enable_rerank)
        print("原因:", result.reason)


if __name__ == "__main__":
    main()