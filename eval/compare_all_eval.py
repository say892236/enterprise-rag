# eval/compare_all_eval.py

"""
四阶段 RAG 评估结果对比脚本。

对比四条链路：
1. Baseline：Retriever → LLM
2. Reranker：Retriever → Reranker → LLM
3. Rewrite + Reranker：Query Rewrite → Retriever → Reranker → LLM
4. Hybrid + Rewrite + Reranker：Query Rewrite → Hybrid Retriever → Reranker → LLM

重点观察：
- Top1 来源是否变化
- Query Rewrite 是否改变了问题
- Hybrid 是否让编号、金额、关键词类问题更稳定
"""

import json
from pathlib import Path


EVAL_FILES = {
    "Baseline": Path("data/eval/baseline_results.jsonl"),
    "Reranker": Path("data/eval/reranker_results.jsonl"),
    "Rewrite+Reranker": Path("data/eval/rewrite_reranker_results.jsonl"),
    "Hybrid+Rewrite+Reranker": Path("data/eval/hybrid_rewrite_reranker_results.jsonl"),
}


def load_jsonl(file_path: Path) -> list[dict]:
    """读取 JSONL 文件，每一行是一条评估结果。"""

    if not file_path.exists():
        raise FileNotFoundError(f"评估文件不存在: {file_path}")

    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def build_question_map(records: list[dict]) -> dict:
    """用 question 作为 key，方便按问题对齐不同评估结果。"""

    return {
        record["question"]: record
        for record in records
    }


def get_top_source(record: dict) -> dict:
    """提取 Top1 来源信息。"""

    top = record.get("top_source")

    if not top:
        return {
            "source": None,
            "header_2": None,
            "chunk_id": None,
            "score": None,
        }

    score = (
        top.get("rerank_score")
        or top.get("hybrid_score")
        or top.get("chroma_score")
        or top.get("score")
    )

    return {
        "source": top.get("source"),
        "header_2": top.get("header_2"),
        "chunk_id": top.get("chunk_id"),
        "score": score,
    }


def print_pipeline_result(name: str, record: dict):
    """打印某条链路的 Top1 结果。"""

    top = get_top_source(record)

    print(
        f"{name:<26} | "
        f"{top['source']} | "
        f"{top['header_2']} | "
        f"{top['chunk_id']} | "
        f"score={top['score']}"
    )


def main():
    # 1. 加载所有评估文件
    eval_maps = {}

    for name, file_path in EVAL_FILES.items():
        records = load_jsonl(file_path)
        eval_maps[name] = build_question_map(records)

    # 2. 以 Baseline 的问题列表作为对比基准
    questions = list(eval_maps["Baseline"].keys())

    print("=" * 140)
    print("四阶段 RAG Top1 对比")
    print("=" * 140)

    for question in questions:
        print("\n问题:", question)

        # 3. 如果 Rewrite 版本存在，打印改写问题
        rewrite_record = eval_maps["Rewrite+Reranker"].get(question)
        hybrid_record = eval_maps["Hybrid+Rewrite+Reranker"].get(question)

        rewritten_question = None

        if hybrid_record:
            rewritten_question = hybrid_record.get("rewritten_question")
        elif rewrite_record:
            rewritten_question = rewrite_record.get("rewritten_question")

        if rewritten_question:
            print("改写:", rewritten_question)

        # 4. 打印四条链路的 Top1
        for name in EVAL_FILES.keys():
            record = eval_maps[name].get(question)

            if record:
                print_pipeline_result(name, record)
            else:
                print(f"{name:<26} | 无结果")

        print("-" * 140)


if __name__ == "__main__":
    main()
