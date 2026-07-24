# run_rag_demo.py

"""
当前最完整 RAG 链路的演示入口。

作用：
1. 使用统一 pipeline 创建 RAG Chain。
2. 输入一个问题。
3. 打印原始问题、改写问题、答案和来源。
"""

from config.settings import HYBRID_TOP_K, RERANK_TOP_K
from pipelines.rag_pipeline import build_hybrid_rag_chain


def print_sources(sources):
    """打印简洁版来源信息。"""

    for i, source in enumerate(sources, start=1):
        print(f"\n【来源 {i}】")
        print("source:", source.get("source"))
        print("header_2:", source.get("header_2"))
        print("chunk_id:", source.get("chunk_id"))
        print("retrieval_sources:", source.get("retrieval_sources"))
        print("hybrid_score:", source.get("hybrid_score"))
        print("rerank_score:", source.get("rerank_score"))


def main():
    # 1. 创建完整 RAG Chain
    rag_chain = build_hybrid_rag_chain()

    # 2. 测试问题
    question = "住宿能报多少？"

    # 3. 执行 RAG 问答
    result = rag_chain.invoke(
        question=question,
        retrieve_top_k=HYBRID_TOP_K,
        rerank_top_k=RERANK_TOP_K,
    )

    # 4. 打印结果
    print("=" * 100)
    print("原始问题：")
    print(result["question"])

    print("=" * 100)
    print("改写问题：")
    print(result.get("rewritten_question"))

    print("=" * 100)
    print("答案：")
    print(result["answer"])

    print("=" * 100)
    print("引用来源：")
    print_sources(result["sources"])


if __name__ == "__main__":
    main()