# eval/compression_ablation_eval.py

"""
Context Compression 消融评测脚本。

作用：
1. 对比开启 / 关闭 Context Compression 的回答结果。
2. 查看压缩前后上下文字符数变化。
3. 查看 LLM 耗时、RAG 总耗时是否有改善。
4. 防止压缩器误删关键制度依据，导致答案质量下降。
"""

import json
import sys
from pathlib import Path


# 把项目根目录加入 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.rag_pipeline import build_hybrid_rag_chain


REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "compression_ablation_report.json"


TEST_CASES = [
    {
        "id": "hotel_standard",
        "question": "员工出差住宿标准是什么？",
        "metadata_filter": {"source": "travel_policy.md"},
    },
    {
        "id": "shanghai_hotel_limit",
        "question": "普通员工去上海出差，酒店最多能报多少？",
        "metadata_filter": {"source": "travel_policy.md"},
    },
    {
        "id": "reimbursement_deadline",
        "question": "员工出差结束后多久内必须提交报销？",
        "metadata_filter": {"source": "travel_policy.md"},
    },
    {
        "id": "five_star_hotel",
        "question": "出差可以住五星级酒店吗？",
        "metadata_filter": {"source": "travel_policy.md"},
    },
    {
        "id": "no_answer_canteen",
        "question": "公司食堂几点开门？",
        "metadata_filter": None,
    },
]


def run_once(chain, case: dict, enable_compression: bool) -> dict:
    """
    执行单次 RAG 问答。
    """

    result = chain.invoke(
        question=case["question"],
        retrieve_top_k=5,
        rerank_top_k=5,
        metadata_filter=case.get("metadata_filter"),
        enable_rewrite=True,
        enable_rerank=True,
        enable_compression=enable_compression,
    )

    return {
        "answer": result.get("answer"),
        "rewritten_question": result.get("rewritten_question"),
        "query_route": result.get("query_route"),
        "compression": result.get("compression"),
        "timings": result.get("timings"),
        "sources": [
            {
                "source": item.get("source"),
                "header_1": item.get("header_1"),
                "header_2": item.get("header_2"),
                "chunk_id": item.get("chunk_id"),
            }
            for item in result.get("sources", [])
        ],
    }


def main():
    """
    主入口。
    """

    print("正在初始化 RAG Chain...")
    chain = build_hybrid_rag_chain()
    print("RAG Chain 初始化完成")

    report = []

    for case in TEST_CASES:
        print("=" * 100)
        print("测试问题:", case["question"])

        print("- 关闭 Context Compression")
        without_compression = run_once(
            chain=chain,
            case=case,
            enable_compression=False,
        )

        print("- 开启 Context Compression")
        with_compression = run_once(
            chain=chain,
            case=case,
            enable_compression=True,
        )

        item = {
            "id": case["id"],
            "question": case["question"],
            "without_compression": without_compression,
            "with_compression": with_compression,
        }

        report.append(item)

        print("关闭压缩答案:", without_compression["answer"])
        print("开启压缩答案:", with_compression["answer"])
        print("压缩信息:", with_compression.get("compression"))
        print("关闭压缩耗时:", without_compression.get("timings"))
        print("开启压缩耗时:", with_compression.get("timings"))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print(f"评测报告已保存到: {REPORT_PATH}")


if __name__ == "__main__":
    main()