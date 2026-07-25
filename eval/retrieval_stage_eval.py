# eval/retrieval_stage_eval.py

"""
RAG 生成前阶段评估。

评估范围：
1. Dense
2. Dense + Reranker
3. Rewrite + Dense + Reranker
4. Rewrite + Hybrid + Reranker

注意：
- 不调用最终回答 LLM。
- Rewrite 阶段本身会调用改写模型，因此 Rewrite 耗时会被统计。
- 输出检索指标和每个阶段的平均耗时。
"""

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable

from config.settings import (
    BM25_TOP_K,
    BM25_WEIGHT,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    HYBRID_TOP_K,
    RERANK_TOP_K,
    RRF_K,
)
from eval.compare_all_eval import (
    compute_hit_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
)
from eval.questions import EVAL_CASES
from pipelines.rag_pipeline import (
    build_bm25_retriever,
    build_dense_retriever,
    build_reranker,
    build_rewriter,
)
from retriever.hybrid_retriever import HybridRetriever


REPORT_JSON = Path(
    "data/eval/retrieval_stage_report.json"
)

REPORT_MD = Path(
    "data/eval/retrieval_stage_report.md"
)


def milliseconds(start_time: float) -> float:
    """将 perf_counter 的耗时转换为毫秒。"""

    return (perf_counter() - start_time) * 1000


def ensure_query_text(value) -> str:
    """
    确保 Query Rewrite 返回字符串。

    避免字典等错误类型进入 Embedding，
    导致 object has no attribute replace。
    """

    if not isinstance(value, str):
        raise TypeError(
            "检索问题必须是字符串，"
            f"当前类型：{type(value).__name__}，"
            f"当前内容：{value!r}"
        )

    value = value.strip()

    if not value:
        raise ValueError("检索问题不能为空")

    return value


def average(values: list[float]) -> float:
    """计算平均值，空列表返回 0。"""

    if not values:
        return 0.0

    return mean(values)


def build_components():
    """
    统一构建评估所需组件。

    Dense Retriever 会同时供：
    - Dense Pipeline
    - Hybrid Pipeline

    这样可以避免重复加载同一个 Embedding 模型。
    """

    dense_retriever = build_dense_retriever()
    bm25_retriever = build_bm25_retriever()
    reranker = build_reranker()
    rewriter = build_rewriter()

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        final_top_k=HYBRID_TOP_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        rrf_k=RRF_K,
    )

    return {
        "dense_retriever": dense_retriever,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "rewriter": rewriter,
    }


def run_dense_pipeline(
    question: str,
    components: dict,
) -> dict:
    """
    Dense Baseline。

    链路：
    Dense Retriever → Top5
    """

    start_total = perf_counter()

    start_retrieve = perf_counter()

    documents = components[
        "dense_retriever"
    ].retrieve(
        query=question,
        top_k=RERANK_TOP_K,
        metadata_filter=None,
    )

    retrieve_ms = milliseconds(start_retrieve)
    total_ms = milliseconds(start_total)

    return {
        "rewritten_question": question,
        "documents": documents[:RERANK_TOP_K],
        "rewrite_ms": 0.0,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": 0.0,
        "total_ms": total_ms,
    }


def run_reranker_pipeline(
    question: str,
    components: dict,
) -> dict:
    """
    Dense + Reranker。

    链路：
    Dense Retriever → Reranker → Top5
    """

    start_total = perf_counter()

    start_retrieve = perf_counter()

    candidates = components[
        "dense_retriever"
    ].retrieve(
        query=question,
        top_k=DENSE_TOP_K,
        metadata_filter=None,
    )

    retrieve_ms = milliseconds(start_retrieve)

    start_rerank = perf_counter()

    documents = components["reranker"].rerank(
        query=question,
        documents=candidates,
        top_k=RERANK_TOP_K,
    )

    rerank_ms = milliseconds(start_rerank)
    total_ms = milliseconds(start_total)

    return {
        "rewritten_question": question,
        "documents": documents,
        "rewrite_ms": 0.0,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "total_ms": total_ms,
    }


def run_rewrite_reranker_pipeline(
    question: str,
    components: dict,
) -> dict:
    """
    Rewrite + Dense + Reranker。

    链路：
    Query Rewrite
    → Dense Retriever
    → Reranker
    → Top5
    """

    start_total = perf_counter()

    start_rewrite = perf_counter()

    rewritten_question = components[
        "rewriter"
    ].rewrite(question)

    rewritten_question = ensure_query_text(
        rewritten_question
    )

    rewrite_ms = milliseconds(start_rewrite)

    start_retrieve = perf_counter()

    candidates = components[
        "dense_retriever"
    ].retrieve(
        query=rewritten_question,
        top_k=DENSE_TOP_K,
        metadata_filter=None,
    )

    retrieve_ms = milliseconds(start_retrieve)

    start_rerank = perf_counter()

    documents = components["reranker"].rerank(
        query=rewritten_question,
        documents=candidates,
        top_k=RERANK_TOP_K,
    )

    rerank_ms = milliseconds(start_rerank)
    total_ms = milliseconds(start_total)

    return {
        "rewritten_question": rewritten_question,
        "documents": documents,
        "rewrite_ms": rewrite_ms,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "total_ms": total_ms,
    }


def run_hybrid_pipeline(
    question: str,
    components: dict,
) -> dict:
    """
    Rewrite + Hybrid + Reranker。

    链路：
    Query Rewrite
    → Dense + BM25 Hybrid
    → Reranker
    → Top5
    """

    start_total = perf_counter()

    start_rewrite = perf_counter()

    rewritten_question = components[
        "rewriter"
    ].rewrite(question)

    rewritten_question = ensure_query_text(
        rewritten_question
    )

    rewrite_ms = milliseconds(start_rewrite)

    start_retrieve = perf_counter()

    candidates = components[
        "hybrid_retriever"
    ].retrieve(
        query=rewritten_question,
        top_k=HYBRID_TOP_K,
        metadata_filter=None,
    )

    retrieve_ms = milliseconds(start_retrieve)

    start_rerank = perf_counter()

    documents = components["reranker"].rerank(
        query=rewritten_question,
        documents=candidates,
        top_k=RERANK_TOP_K,
    )

    rerank_ms = milliseconds(start_rerank)
    total_ms = milliseconds(start_total)

    return {
        "rewritten_question": rewritten_question,
        "documents": documents,
        "rewrite_ms": rewrite_ms,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "total_ms": total_ms,
    }


PIPELINES: dict[str, Callable] = {
    "Dense": run_dense_pipeline,
    "Dense+Reranker": run_reranker_pipeline,
    "Rewrite+Dense+Reranker": (
        run_rewrite_reranker_pipeline
    ),
    "Rewrite+Hybrid+Reranker": (
        run_hybrid_pipeline
    ),
}


def documents_to_sources(
    documents,
) -> list[dict]:
    """
    将 Document 转成指标函数需要的 source 结构。
    """

    return [
        dict(document.metadata or {})
        for document in documents
    ]


def evaluate_pipeline(
    pipeline_name: str,
    pipeline_func: Callable,
    components: dict,
) -> tuple[dict, list[dict]]:
    """
    执行一条 Pipeline 的全部评估用例。
    """

    hit_at_1_values = []
    hit_at_3_values = []
    hit_at_5_values = []
    recall_at_5_values = []
    precision_at_5_values = []
    mrr_values = []

    rewrite_times = []
    retrieve_times = []
    rerank_times = []
    total_times = []

    case_results = []

    qa_cases = [
        case
        for case in EVAL_CASES
        if case["case_type"] == "qa"
    ]

    for index, case in enumerate(
        qa_cases,
        start=1,
    ):
        question = str(case["question"])
        expected_targets = case["expected_targets"]

        print(
            f"[{pipeline_name}] "
            f"{index}/{len(qa_cases)} "
            f"{question}"
        )

        result = pipeline_func(
            question,
            components,
        )

        sources = documents_to_sources(
            result["documents"]
        )

        hit_at_1 = compute_hit_at_k(
            sources,
            expected_targets,
            1,
        )

        hit_at_3 = compute_hit_at_k(
            sources,
            expected_targets,
            3,
        )

        hit_at_5 = compute_hit_at_k(
            sources,
            expected_targets,
            5,
        )

        recall_at_5 = compute_recall_at_k(
            sources,
            expected_targets,
            5,
        )

        precision_at_5 = compute_precision_at_k(
            sources,
            expected_targets,
            5,
        )

        mrr = compute_mrr(
            sources,
            expected_targets,
        )

        hit_at_1_values.append(hit_at_1)
        hit_at_3_values.append(hit_at_3)
        hit_at_5_values.append(hit_at_5)
        recall_at_5_values.append(recall_at_5)
        precision_at_5_values.append(
            precision_at_5
        )
        mrr_values.append(mrr)

        rewrite_times.append(result["rewrite_ms"])
        retrieve_times.append(result["retrieve_ms"])
        rerank_times.append(result["rerank_ms"])
        total_times.append(result["total_ms"])

        case_results.append(
            {
                "id": case["id"],
                "question": question,
                "rewritten_question": result[
                    "rewritten_question"
                ],
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "recall_at_5": recall_at_5,
                "precision_at_5": precision_at_5,
                "mrr": mrr,
                "rewrite_ms": round(
                    result["rewrite_ms"],
                    2,
                ),
                "retrieve_ms": round(
                    result["retrieve_ms"],
                    2,
                ),
                "rerank_ms": round(
                    result["rerank_ms"],
                    2,
                ),
                "total_ms": round(
                    result["total_ms"],
                    2,
                ),
                "sources": sources,
            }
        )

    summary = {
        "evaluated_cases": len(qa_cases),
        "hit_at_1": average(hit_at_1_values),
        "hit_at_3": average(hit_at_3_values),
        "hit_at_5": average(hit_at_5_values),
        "recall_at_5": average(
            recall_at_5_values
        ),
        "precision_at_5": average(
            precision_at_5_values
        ),
        "mrr": average(mrr_values),
        "average_rewrite_ms": average(
            rewrite_times
        ),
        "average_retrieve_ms": average(
            retrieve_times
        ),
        "average_rerank_ms": average(
            rerank_times
        ),
        "average_total_ms": average(
            total_times
        ),
    }

    return summary, case_results


def format_percent(value: float) -> str:
    """格式化百分比。"""

    return f"{value * 100:.2f}%"


def print_summary(
    summaries: dict[str, dict],
) -> None:
    """打印最终对比结果。"""

    print("=" * 165)
    print("RAG 生成前阶段定量评估")
    print("=" * 165)

    print(
        f"{'Pipeline':<30}"
        f"{'Hit@1':>11}"
        f"{'Recall@5':>13}"
        f"{'Precision@5':>15}"
        f"{'MRR':>10}"
        f"{'Rewrite':>15}"
        f"{'Retrieve':>15}"
        f"{'Rerank':>15}"
        f"{'总耗时':>15}"
    )

    print("-" * 165)

    for name, summary in summaries.items():
        print(
            f"{name:<30}"
            f"{format_percent(summary['hit_at_1']):>11}"
            f"{format_percent(summary['recall_at_5']):>13}"
            f"{format_percent(summary['precision_at_5']):>15}"
            f"{summary['mrr']:>10.4f}"
            f"{summary['average_rewrite_ms']:>12.2f} ms"
            f"{summary['average_retrieve_ms']:>12.2f} ms"
            f"{summary['average_rerank_ms']:>12.2f} ms"
            f"{summary['average_total_ms']:>12.2f} ms"
        )

    print("=" * 165)
    print("说明：总耗时不包含最终答案生成。")


def save_reports(
    summaries: dict[str, dict],
    details: dict[str, list[dict]],
) -> None:
    """保存 JSON 和 Markdown 报告。"""

    REPORT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "scope": (
            "pre-generation stages only; "
            "final answer generation excluded"
        ),
        "summaries": summaries,
        "details": details,
    }

    with open(
        REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# RAG 生成前阶段定量评估",
        "",
        "| Pipeline | Hit@1 | Recall@5 | Precision@5 | MRR | Rewrite | Retrieve | Rerank | 总耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, summary in summaries.items():
        lines.append(
            f"| {name} "
            f"| {format_percent(summary['hit_at_1'])} "
            f"| {format_percent(summary['recall_at_5'])} "
            f"| {format_percent(summary['precision_at_5'])} "
            f"| {summary['mrr']:.4f} "
            f"| {summary['average_rewrite_ms']:.2f} ms "
            f"| {summary['average_retrieve_ms']:.2f} ms "
            f"| {summary['average_rerank_ms']:.2f} ms "
            f"| {summary['average_total_ms']:.2f} ms |"
        )

    lines.extend(
        [
            "",
            "> 总耗时不包含最终答案生成。",
            "",
            "> Rewrite 阶段可能调用外部大模型，因此会受到网络和模型服务波动影响。",
        ]
    )

    with open(
        REPORT_MD,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))

    print(f"JSON 报告：{REPORT_JSON}")
    print(f"Markdown 报告：{REPORT_MD}")


def main():
    components = build_components()

    summaries = {}
    details = {}

    for name, pipeline_func in PIPELINES.items():
        print("\n" + "=" * 100)
        print(f"开始评估：{name}")
        print("=" * 100)

        summary, case_results = evaluate_pipeline(
            pipeline_name=name,
            pipeline_func=pipeline_func,
            components=components,
        )

        summaries[name] = summary
        details[name] = case_results

    print_summary(summaries)

    save_reports(
        summaries=summaries,
        details=details,
    )


if __name__ == "__main__":
    main()