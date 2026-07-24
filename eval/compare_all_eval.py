# eval/compare_all_eval.py

"""
四阶段 RAG 定量对比评估。

对比链路：
1. Baseline
2. Reranker
3. Rewrite + Reranker
4. Hybrid + Rewrite + Reranker

统计指标：
- Hit@1
- Hit@3
- Hit@5
- Recall@5
- Precision@5
- MRR
- 平均端到端耗时

相关结果使用 source + header_2 判断。
不使用 chunk_id，因为数据库重建后 chunk_id 可能变化。
"""

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Optional

from eval.questions import EVAL_CASES


EVAL_FILES = {
    "Baseline": Path("data/eval/baseline_results.jsonl"),
    "Reranker": Path("data/eval/reranker_results.jsonl"),
    "Rewrite+Reranker": Path(
        "data/eval/rewrite_reranker_results.jsonl"
    ),
    "Hybrid+Rewrite+Reranker": Path(
        "data/eval/hybrid_rewrite_reranker_results.jsonl"
    ),
}

REPORT_JSON = Path(
    "data/eval/pipeline_comparison_report.json"
)

REPORT_MD = Path(
    "data/eval/pipeline_comparison_report.md"
)


def load_jsonl(file_path: Path) -> list[dict]:
    """读取 JSONL 文件。"""

    if not file_path.exists():
        raise FileNotFoundError(
            f"评估文件不存在：{file_path}"
        )

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def normalize_text(value: Optional[str]) -> str:
    """
    标准化字符串，减少空格差异对匹配的影响。
    """

    if value is None:
        return ""

    return " ".join(str(value).split()).strip()


def target_matches_source(
    target: dict,
    source: dict,
) -> bool:
    """
    判断一条检索结果是否匹配标准相关目标。

    匹配规则：
    1. source 必须一致。
    2. expected header_2 为 None 时，只判断 source。
    3. 否则 source 和 header_2 都必须一致。
    """

    expected_source = normalize_text(
        target.get("source")
    )

    actual_source = normalize_text(
        source.get("source")
    )

    if expected_source != actual_source:
        return False

    expected_header = target.get("header_2")

    # 制度编号等内容可能位于文档开头，
    # 没有稳定的二级标题，此时只判断文档来源。
    if expected_header is None:
        return True

    actual_header = source.get("header_2")

    return normalize_text(expected_header) == normalize_text(
        actual_header
    )


def source_is_relevant(
    source: dict,
    expected_targets: list[dict],
) -> bool:
    """判断检索结果是否属于任意一个相关目标。"""

    return any(
        target_matches_source(target, source)
        for target in expected_targets
    )


def compute_hit_at_k(
    sources: list[dict],
    expected_targets: list[dict],
    k: int,
) -> float:
    """
    前 K 条中只要出现至少一个相关结果，就记为命中。
    """

    top_k_sources = sources[:k]

    return float(
        any(
            source_is_relevant(
                source,
                expected_targets,
            )
            for source in top_k_sources
        )
    )


def compute_recall_at_k(
    sources: list[dict],
    expected_targets: list[dict],
    k: int,
) -> float:
    """
    Recall@K：
        TopK 中找回的相关目标数量
        ÷
        所有标准相关目标数量

    例如某问题有两个相关章节，
    Top5 只找回其中一个，则 Recall@5 = 50%。
    """

    if not expected_targets:
        return 0.0

    matched_target_indexes = set()

    for source in sources[:k]:
        for index, target in enumerate(
            expected_targets
        ):
            if target_matches_source(
                target,
                source,
            ):
                matched_target_indexes.add(index)

    return (
        len(matched_target_indexes)
        / len(expected_targets)
    )


def compute_precision_at_k(
    sources: list[dict],
    expected_targets: list[dict],
    k: int,
) -> float:
    """
    Precision@K：
        TopK 中的相关结果数量
        ÷
        K

    即使 sources 少于 K 条，分母仍使用 K。
    """

    relevant_count = sum(
        1
        for source in sources[:k]
        if source_is_relevant(
            source,
            expected_targets,
        )
    )

    return relevant_count / k


def compute_mrr(
    sources: list[dict],
    expected_targets: list[dict],
) -> float:
    """
    MRR：
    第一个相关结果排名的倒数。

    第1名相关：1.0
    第2名相关：0.5
    第3名相关：0.3333
    没有相关结果：0
    """

    for rank, source in enumerate(
        sources,
        start=1,
    ):
        if source_is_relevant(
            source,
            expected_targets,
        ):
            return 1.0 / rank

    return 0.0


def build_record_map(
    records: list[dict],
) -> dict[str, dict]:
    """使用 question 对齐评估结果。"""

    return {
        record["question"]: record
        for record in records
    }


def average(values: list[float]) -> Optional[float]:
    """计算平均值，空列表返回 None。"""

    if not values:
        return None

    return mean(values)


def evaluate_pipeline(
    records: list[dict],
) -> tuple[dict, list[str]]:
    """
    对单条 Pipeline 计算全部指标。

    no_answer 用例不参与检索指标，
    因为它们没有标准相关文档。
    """

    record_map = build_record_map(records)

    hit_at_1_values = []
    hit_at_3_values = []
    hit_at_5_values = []
    recall_at_5_values = []
    precision_at_5_values = []
    mrr_values = []
    latency_values = []

    missing_questions = []

    qa_cases = [
        case
        for case in EVAL_CASES
        if case["case_type"] == "qa"
    ]

    for case in qa_cases:
        question = case["question"]
        expected_targets = case["expected_targets"]

        record = record_map.get(question)

        if not record:
            missing_questions.append(question)
            continue

        sources = record.get("sources") or []

        hit_at_1_values.append(
            compute_hit_at_k(
                sources,
                expected_targets,
                1,
            )
        )

        hit_at_3_values.append(
            compute_hit_at_k(
                sources,
                expected_targets,
                3,
            )
        )

        hit_at_5_values.append(
            compute_hit_at_k(
                sources,
                expected_targets,
                5,
            )
        )

        recall_at_5_values.append(
            compute_recall_at_k(
                sources,
                expected_targets,
                5,
            )
        )

        precision_at_5_values.append(
            compute_precision_at_k(
                sources,
                expected_targets,
                5,
            )
        )

        mrr_values.append(
            compute_mrr(
                sources,
                expected_targets,
            )
        )

        latency_ms = record.get("latency_ms")

        if latency_ms is not None:
            latency_values.append(
                float(latency_ms)
            )

    return {
        "evaluated_qa_cases": len(
            hit_at_1_values
        ),
        "hit_at_1": average(
            hit_at_1_values
        ),
        "hit_at_3": average(
            hit_at_3_values
        ),
        "hit_at_5": average(
            hit_at_5_values
        ),
        "recall_at_5": average(
            recall_at_5_values
        ),
        "precision_at_5": average(
            precision_at_5_values
        ),
        "mrr": average(
            mrr_values
        ),
        "average_latency_ms": average(
            latency_values
        ),
    }, missing_questions


def format_percent(
    value: Optional[float],
) -> str:
    """格式化百分比。"""

    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def format_number(
    value: Optional[float],
    digits: int = 4,
) -> str:
    """格式化普通数值。"""

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def format_latency(
    value: Optional[float],
) -> str:
    """格式化毫秒耗时。"""

    if value is None:
        return "N/A"

    return f"{value:.2f} ms"


def print_summary(
    summaries: dict[str, dict],
):
    """打印四阶段指标对比表。"""

    print("=" * 155)
    print("四阶段 RAG 定量评估")
    print("=" * 155)

    header = (
        f"{'Pipeline':<30}"
        f"{'Hit@1':>12}"
        f"{'Hit@3':>12}"
        f"{'Hit@5':>12}"
        f"{'Recall@5':>14}"
        f"{'Precision@5':>16}"
        f"{'MRR':>12}"
        f"{'平均端到端耗时':>22}"
    )

    print(header)
    print("-" * 155)

    for name, summary in summaries.items():
        print(
            f"{name:<30}"
            f"{format_percent(summary['hit_at_1']):>12}"
            f"{format_percent(summary['hit_at_3']):>12}"
            f"{format_percent(summary['hit_at_5']):>12}"
            f"{format_percent(summary['recall_at_5']):>14}"
            f"{format_percent(summary['precision_at_5']):>16}"
            f"{format_number(summary['mrr']):>12}"
            f"{format_latency(summary['average_latency_ms']):>22}"
        )

    print("=" * 155)
    print("说明：no_answer 用例不参与检索指标计算。")
    print("耗时为完整 RAG 请求耗时，不是纯 Retriever 耗时。")


def save_reports(
    summaries: dict[str, dict],
):
    """保存 JSON 和 Markdown 对比报告。"""

    REPORT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = {
        "generated_at": datetime.now().isoformat(),
        "metric_scope": (
            "qa cases only; no_answer cases excluded"
        ),
        "latency_scope": (
            "end-to-end RAG latency, including LLM"
        ),
        "pipelines": summaries,
    }

    with open(
        REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# 四阶段 RAG 定量评估",
        "",
        "| Pipeline | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR | 平均端到端耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, summary in summaries.items():
        lines.append(
            f"| {name} "
            f"| {format_percent(summary['hit_at_1'])} "
            f"| {format_percent(summary['hit_at_3'])} "
            f"| {format_percent(summary['hit_at_5'])} "
            f"| {format_percent(summary['recall_at_5'])} "
            f"| {format_percent(summary['precision_at_5'])} "
            f"| {format_number(summary['mrr'])} "
            f"| {format_latency(summary['average_latency_ms'])} |"
        )

    lines.extend(
        [
            "",
            "> no_answer 用例不参与检索指标计算。",
            "",
            "> 耗时为完整 RAG 请求耗时，包含检索、改写、重排和 LLM 生成。",
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
    summaries = {}

    for name, file_path in EVAL_FILES.items():
        records = load_jsonl(file_path)

        summary, missing_questions = evaluate_pipeline(
            records
        )

        summaries[name] = summary

        if missing_questions:
            print(
                f"[WARN] {name} 缺少 "
                f"{len(missing_questions)} 条评估结果："
            )

            for question in missing_questions:
                print(f"  - {question}")

    print_summary(summaries)
    save_reports(summaries)


if __name__ == "__main__":
    main()