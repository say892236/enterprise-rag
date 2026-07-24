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
            f"chroma_score={source.get('chroma_score')}"
        )

    print()


def run_eval(
    pipeline_name: str,
    rag_chain,
    questions: List[str],
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

    # 每次评估前删除旧文件，避免重复追加
    if output_path.exists():
        output_path.unlink()

    for index, question in enumerate(questions, start=1):
        # 执行一次 RAG 问答
        result = rag_chain.invoke(
            question=question,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
        )

        # 控制台打印
        print_result(
            index=index,
            total=len(questions),
            result=result,
        )

        sources = result["sources"]

        # 保存结构化结果，方便后续 compare 脚本读取
        record = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": pipeline_name,
            "question": result["question"],
            "rewritten_question": result.get("rewritten_question"),
            "answer": result["answer"],
            "sources": sources,
            "top_source": sources[0] if sources else None,
            "retrieve_top_k": retrieve_top_k,
            "rerank_top_k": rerank_top_k,
        }

        save_jsonl(output_path, record)

    print("\n评估完成")
    print("pipeline:", pipeline_name)
    print("结果保存:", output_path)