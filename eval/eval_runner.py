# eval/eval_runner.py

"""
统一评估执行器。

作用：
1. 接收一个 RAG Chain。
2. 批量执行 TEST_QUESTIONS。
3. 打印结果。
4. 保存 JSONL。

这样 baseline_eval.py、reranker_eval.py 等文件就不用重复写一堆循环逻辑。
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List
from time import perf_counter

from config.settings import HYBRID_TOP_K, RERANK_TOP_K


def save_jsonl(output_file: Path, data: dict):
    """
    保存一条 JSONL 数据。

    JSONL 特点：
    - 一行一个 JSON
    - 适合保存批量评估结果
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def print_result(index: int, total: int, result: dict):
    """
    控制台打印单条评估结果。
    """

    print("=" * 120)
    print(f"测试 {index}/{total}")

    print("\n原始问题:")
    print(result["question"])

    if result.get("rewritten_question"):
        print("\n改写问题:")
        print(result["rewritten_question"])

    print("\n答案:")
    print(result["answer"])

    if result.get("latency_ms") is not None:
        print(f"\n端到端耗时: {result['latency_ms']:.2f} ms")

    print("\nTop Sources:")

    for i, source in enumerate(result["sources"], start=1):
        print(
            f"{i}. "
            f"source={source.get('source')} | "
            f"header_2={source.get('header_2')} | "
            f"chunk_id={source.get('chunk_id')} | "
            f"retrieval_sources={source.get('retrieval_sources')} | "
            f"hybrid_score={source.get('hybrid_score')} | "
            f"rerank_score={source.get('rerank_score')} | "
            f"dense_score={source.get('milvus_score') or source.get('score')}"
        )

    print()


def run_eval(
    pipeline_name: str,
    rag_chain,
    questions: List[str | dict],
    output_file: str,
    retrieve_top_k: int = HYBRID_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
):
    """
    统一执行评估。

    参数：
    pipeline_name:
        当前评估链路名称，例如 baseline / reranker / hybrid。

    rag_chain:
        已构建好的 EnterpriseRAGChain。

    questions:
        测试问题列表。

    output_file:
        JSONL 保存路径。

    retrieve_top_k:
        Retriever 召回数量。

    rerank_top_k:
        Reranker 保留数量。
    """

    output_path = Path(output_file)

    # 防止问题集为空时误打印“评估完成”。
    if not questions:
        raise ValueError(
            "评估问题集为空，请检查 eval/questions.py 中的 "
            "EVAL_CASES 和 TEST_QUESTIONS。"
        )

    # 每次评估前删除旧结果，避免重复追加。
    if output_path.exists():
        output_path.unlink()

    # 注意：for 循环必须和上面的 if 保持同一级缩进。
    for index, question_item in enumerate(
        questions,
        start=1,
    ):
        # 兼容两种问题格式：
        # 1. 字符串
        # 2. 包含 question 字段的字典
        if isinstance(question_item, dict):
            question = question_item.get("question")
        else:
            question = question_item

        # Retriever 和 Embedding 只能接收字符串。
        if not isinstance(question, str):
            raise TypeError(
                "评估问题必须是字符串，"
                f"当前类型：{type(question).__name__}，"
                f"当前内容：{question!r}"
            )

        question = question.strip()

        if not question:
            raise ValueError(
                f"第 {index} 条评估问题为空"
            )

        # 记录完整 RAG 链路的端到端耗时。
        start_time = perf_counter()

        result = rag_chain.invoke(
            question=question,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
        )

        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        # 写入 result，方便控制台打印。
        result["latency_ms"] = latency_ms

        # 控制台打印当前这一条结果。
        print_result(
            index=index,
            total=len(questions),
            result=result,
        )

        sources = result.get("sources") or []

        # 保存当前这一条结构化评估结果。
        record = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": pipeline_name,
            "question": result["question"],
            "rewritten_question": result.get(
                "rewritten_question"
            ),
            "answer": result["answer"],
            "sources": sources,
            "top_source": (
                sources[0]
                if sources
                else None
            ),
            "retrieve_top_k": retrieve_top_k,
            "rerank_top_k": rerank_top_k,

            # 完整 RAG 请求耗时，不是纯检索耗时。
            "latency_ms": round(
                latency_ms,
                2,
            ),
        }

        save_jsonl(
            output_file=output_path,
            data=record,
        )

    print("\n评估完成")
    print("pipeline:", pipeline_name)
    print("结果保存:", output_path)