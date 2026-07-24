# eval/final_rag_eval.py

"""
Enterprise-RAG 最终 API 评估脚本。

评估目标：
1. 检索命中率 Hit@K
2. 检索准确率 Precision@K
3. 排序质量 MRR
4. 答案关键事实覆盖率
5. 简化版忠实度 Faithfulness
6. 无答案拒答准确率
7. 权限拦截准确率
8. API 响应时间统计

运行前提：
1. Docker Compose API 已启动：
   docker compose up -d

2. Milvus 已完成入库：
   python build_database.py

3. API 可访问：
   http://127.0.0.1:8000

运行方式：
python -m eval.final_rag_eval
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# API 地址，默认使用本机 Docker Compose 暴露端口
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# 测试账号：沿用当前项目里的普通员工账号
USERNAME = os.getenv("EVAL_USERNAME", "ay")
PASSWORD = os.getenv("EVAL_PASSWORD", "123456")

# 报告输出目录
REPORT_DIR = Path("data/eval")
REPORT_JSON_PATH = REPORT_DIR / "final_rag_eval_report.json"
REPORT_MD_PATH = REPORT_DIR / "final_rag_eval_report.md"


@dataclass
class EvalCase:
    """
    单条 RAG 评估用例。

    字段说明：
        case_id: 用例 ID
        question: 用户问题
        expected_sources: 期望命中的文档 source
        expected_keywords: 答案中应该包含的关键事实
        case_type: qa / no_answer / permission
        metadata_filter: 可选元数据过滤条件
    """

    case_id: str
    question: str
    expected_sources: List[str]
    expected_keywords: List[str]
    case_type: str = "qa"
    metadata_filter: Optional[Dict[str, Any]] = None


@dataclass
class EvalResult:
    """
    单条用例评估结果。
    """

    case_id: str
    question: str
    case_type: str

    status_code: int
    latency_ms: float

    answer: str
    sources: List[str]

    hit_at_1: Optional[float]
    hit_at_3: Optional[float]
    hit_at_5: Optional[float]
    precision_at_5: Optional[float]
    mrr: Optional[float]

    fact_coverage: Optional[float]
    faithfulness: Optional[float]

    no_answer_correct: Optional[float]
    permission_correct: Optional[float]

    passed: bool
    error: Optional[str] = None


def print_step(message: str) -> None:
    """
    打印当前评估步骤。
    """

    print(f"\n[STEP] {message}")


def ensure_report_dir() -> None:
    """
    确保报告目录存在。
    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def request_json(response: requests.Response) -> Dict[str, Any]:
    """
    将 HTTP 响应解析为 JSON。
    """

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"响应不是合法 JSON，status_code={response.status_code}, text={response.text}"
        ) from exc


def check_health() -> None:
    """
    检查 API 是否可用。
    """

    print_step("检查 /health")

    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"/health 异常，status_code={response.status_code}, body={response.text}"
        )

    print("[PASS] /health 正常")


def login() -> str:
    """
    登录并返回 JWT access_token。
    """

    print_step("登录获取 Token")

    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json={
            "username": USERNAME,
            "password": PASSWORD,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"登录失败，status_code={response.status_code}, body={response.text}"
        )

    data = request_json(response)

    # 兼容统一响应格式：
    # 1. {"access_token": "..."}
    # 2. {"success": true, "data": {"access_token": "..."}}
    if isinstance(data.get("data"), dict):
        token = data["data"].get("access_token")
    else:
        token = data.get("access_token")

    if not token:
        raise RuntimeError(f"登录响应中没有 access_token：{data}")

    print("[PASS] 登录成功")

    return token


def call_chat(
    token: str,
    question: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> tuple[int, Dict[str, Any], float]:
    """
    调用 /chat 接口，并返回状态码、JSON 数据、耗时。

    返回：
        status_code
        response_json
        latency_ms
    """

    body: Dict[str, Any] = {
        "question": question,
        "retrieve_top_k": 10,
        "rerank_top_k": 5,
    }

    if metadata_filter:
        body["metadata_filter"] = metadata_filter

    start_time = time.perf_counter()

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json=body,
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=180,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    try:
        data = request_json(response)
    except RuntimeError:
        data = {
            "raw_text": response.text,
        }

    return response.status_code, data, latency_ms


def extract_chat_data(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 API 响应中提取 data 部分。

    兼容两种格式：
    1. {"success": true, "data": {...}}
    2. {...}
    """

    if isinstance(response_json.get("data"), dict):
        return response_json["data"]

    return response_json


def normalize_source_name(source: Any) -> str:
    """
    从 source 对象中提取 source 文件名。

    兼容多种可能结构：
    1. "travel_policy.md"
    2. {"source": "travel_policy.md"}
    3. {"metadata": {"source": "travel_policy.md"}}
    """

    if isinstance(source, str):
        return source

    if not isinstance(source, dict):
        return str(source)

    if source.get("source"):
        return str(source["source"])

    metadata = source.get("metadata")
    if isinstance(metadata, dict) and metadata.get("source"):
        return str(metadata["source"])

    return str(source)


def extract_source_names(chat_data: Dict[str, Any]) -> List[str]:
    """
    从 /chat 响应中提取 sources 文件名列表。
    """

    raw_sources = chat_data.get("sources", [])

    if not isinstance(raw_sources, list):
        return []

    return [normalize_source_name(item) for item in raw_sources]


def extract_source_text(chat_data: Dict[str, Any]) -> str:
    """
    尝试从 sources 中提取检索上下文文本。

    支持字段：
    - content
    - page_content
    - content_preview
    - text
    - chunk
    - chunk_text
    """

    raw_sources = chat_data.get("sources", [])

    if not isinstance(raw_sources, list):
        return ""

    text_parts: List[str] = []

    for item in raw_sources:
        if not isinstance(item, dict):
            continue

        for key in [
            "content",
            "page_content",
            "content_preview",
            "text",
            "chunk",
            "chunk_text",
        ]:
            value = item.get(key)

            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
                break

    return "\n".join(text_parts)

def compute_hit_at_k(
    sources: List[str],
    expected_sources: List[str],
    k: int,
) -> float:
    """
    计算 Hit@K。

    只要 TopK 中出现任意 expected_source，就记为 1。
    """

    top_k_sources = sources[:k]

    for source in top_k_sources:
        for expected in expected_sources:
            if expected in source:
                return 1.0

    return 0.0


def compute_precision_at_k(
    sources: List[str],
    expected_sources: List[str],
    k: int,
) -> float:
    """
    计算 Precision@K。

    当前按 source 文件级别判断相关性：
    TopK 中 source 命中 expected_sources 即认为相关。
    """

    top_k_sources = sources[:k]

    if not top_k_sources:
        return 0.0

    relevant_count = 0

    for source in top_k_sources:
        if any(expected in source for expected in expected_sources):
            relevant_count += 1

    return relevant_count / len(top_k_sources)


def compute_mrr(
    sources: List[str],
    expected_sources: List[str],
) -> float:
    """
    计算 MRR。

    找到第一个相关 source 的排名 rank：
        MRR = 1 / rank

    如果没有命中，返回 0。
    """

    for index, source in enumerate(sources, start=1):
        if any(expected in source for expected in expected_sources):
            return 1.0 / index

    return 0.0


def compute_fact_coverage(
    answer: str,
    expected_keywords: List[str],
) -> float:
    """
    计算答案关键事实覆盖率。

    expected_keywords 中有多少关键事实出现在 answer 中。
    """

    if not expected_keywords:
        return 1.0

    hit_count = 0

    for keyword in expected_keywords:
        if keyword in answer:
            hit_count += 1

    return hit_count / len(expected_keywords)


def compute_simple_faithfulness(
    answer: str,
    source_text: str,
    expected_keywords: List[str],
) -> Optional[float]:
    """
    简化版忠实度评估。

    判断 expected_keywords 中出现在答案里的事实，
    是否也能在检索上下文 source_text 中找到。

    返回：
        1.0：答案中的关键事实都能在上下文找到
        0.0：存在答案关键事实无法从上下文支持
        None：没有 source_text，无法判断

    注意：
        这是规则版 faithfulness，不是 LLM-as-Judge。
        后续可以升级为大模型裁判。
    """

    if not source_text.strip():
        return None

    answer_facts = [
        keyword for keyword in expected_keywords
        if keyword in answer
    ]

    if not answer_facts:
        return 0.0

    supported_count = 0

    for fact in answer_facts:
        if fact in source_text:
            supported_count += 1

    return supported_count / len(answer_facts)


def build_eval_cases() -> List[EvalCase]:
    """
    构造最终评估集。

    注意：
    这里先使用当前企业制度知识库中的典型问题。
    后续可以继续扩充到 30~50 条。
    """

    return [
        EvalCase(
            case_id="qa_travel_hotel_standard",
            question="员工出差住宿标准是什么？",
            expected_sources=["travel_policy.md"],
            expected_keywords=["一类城市", "二类城市", "三类城市", "600", "450", "350"],
        ),
        EvalCase(
            case_id="qa_shanghai_hotel_limit",
            question="普通员工去上海出差，酒店最多能报多少？",
            expected_sources=["travel_policy.md"],
            expected_keywords=["上海", "一类城市", "普通员工", "600"],
        ),
        EvalCase(
            case_id="qa_reimbursement_deadline",
            question="员工出差结束后多久内必须提交报销？",
            expected_sources=["travel_policy.md", "reimbursement_policy.md"],
            expected_keywords=["10", "工作日", "报销"],
        ),
        EvalCase(
            case_id="qa_five_star_hotel",
            question="出差可以住五星级酒店吗？",
            expected_sources=["travel_policy.md"],
            expected_keywords=["五星级", "住宿标准", "审批"],
        ),
        EvalCase(
            case_id="qa_leave_policy",
            question="员工请假需要提前多久申请？",
            expected_sources=["leave_policy.md"],
            expected_keywords=["请假", "提前", "申请"],
        ),
        EvalCase(
            case_id="qa_reimbursement_invoice",
            question="报销时需要提供发票吗？",
            expected_sources=["reimbursement_policy.md"],
            expected_keywords=["报销", "发票"],
        ),
        EvalCase(
            case_id="no_answer_canteen",
            question="公司食堂几点开门？",
            expected_sources=[],
            expected_keywords=[],
            case_type="no_answer",
        ),
        EvalCase(
            case_id="no_answer_parking_fee",
            question="公司停车场怎么收费？",
            expected_sources=[],
            expected_keywords=[],
            case_type="no_answer",
        ),
        EvalCase(
            case_id="permission_hr_salary",
            question="HR 私有薪酬制度有哪些内容？",
            expected_sources=["hr_salary_policy.md"],
            expected_keywords=[],
            case_type="permission",
            metadata_filter={
                "source": "hr_salary_policy.md",
            },
        ),
    ]


def evaluate_qa_case(
    case: EvalCase,
    status_code: int,
    response_json: Dict[str, Any],
    latency_ms: float,
) -> EvalResult:
    """
    评估普通 RAG 问答用例。
    """

    chat_data = extract_chat_data(response_json)

    answer = str(chat_data.get("answer", ""))
    sources = extract_source_names(chat_data)
    source_text = extract_source_text(chat_data)


    hit_at_1 = compute_hit_at_k(sources, case.expected_sources, 1)
    hit_at_3 = compute_hit_at_k(sources, case.expected_sources, 3)
    hit_at_5 = compute_hit_at_k(sources, case.expected_sources, 5)
    precision_at_5 = compute_precision_at_k(sources, case.expected_sources, 5)
    mrr = compute_mrr(sources, case.expected_sources)

    fact_coverage = compute_fact_coverage(answer, case.expected_keywords)
    faithfulness = compute_simple_faithfulness(
        answer=answer,
        source_text=source_text,
        expected_keywords=case.expected_keywords,
    )

    passed = (
        status_code == 200
        and hit_at_5 == 1.0
        and fact_coverage >= 0.5
    )

    error = None

    if status_code != 200:
        error = f"状态码异常：{status_code}"
    elif hit_at_5 != 1.0:
        error = f"Top5 未命中期望文档：expected={case.expected_sources}, actual={sources}"
    elif fact_coverage < 0.5:
        error = f"关键事实覆盖率过低：{fact_coverage}"

    return EvalResult(
        case_id=case.case_id,
        question=case.question,
        case_type=case.case_type,

        status_code=status_code,
        latency_ms=latency_ms,

        answer=answer,
        sources=sources,

        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        precision_at_5=precision_at_5,
        mrr=mrr,

        fact_coverage=fact_coverage,
        faithfulness=faithfulness,

        no_answer_correct=None,
        permission_correct=None,

        passed=passed,
        error=error,
    )


def evaluate_no_answer_case(
    case: EvalCase,
    status_code: int,
    response_json: Dict[str, Any],
    latency_ms: float,
) -> EvalResult:
    """
    评估无答案拒答用例。
    """

    chat_data = extract_chat_data(response_json)

    answer = str(chat_data.get("answer", ""))
    sources = extract_source_names(chat_data)

    expected_answer = "知识库中未找到相关信息。"

    no_answer_correct = 1.0 if (
        status_code == 200
        and answer == expected_answer
        and len(sources) == 0
    ) else 0.0

    return EvalResult(
        case_id=case.case_id,
        question=case.question,
        case_type=case.case_type,

        status_code=status_code,
        latency_ms=latency_ms,

        answer=answer,
        sources=sources,

        hit_at_1=None,
        hit_at_3=None,
        hit_at_5=None,
        precision_at_5=None,
        mrr=None,

        fact_coverage=None,
        faithfulness=None,

        no_answer_correct=no_answer_correct,
        permission_correct=None,

        passed=no_answer_correct == 1.0,
        error=None if no_answer_correct == 1.0 else f"无答案拒答异常：answer={answer}, sources={sources}",
    )


def evaluate_permission_case(
    case: EvalCase,
    status_code: int,
    response_json: Dict[str, Any],
    latency_ms: float,
) -> EvalResult:
    """
    评估权限拦截用例。
    """

    permission_correct = 0.0

    error_code = response_json.get("error_code")

    if status_code == 403 and error_code == "PERMISSION_DENIED":
        permission_correct = 1.0

    return EvalResult(
        case_id=case.case_id,
        question=case.question,
        case_type=case.case_type,

        status_code=status_code,
        latency_ms=latency_ms,

        answer="",
        sources=[],

        hit_at_1=None,
        hit_at_3=None,
        hit_at_5=None,
        precision_at_5=None,
        mrr=None,

        fact_coverage=None,
        faithfulness=None,

        no_answer_correct=None,
        permission_correct=permission_correct,

        passed=permission_correct == 1.0,
        error=None if permission_correct == 1.0 else f"权限拦截异常：status={status_code}, response={response_json}",
    )


def evaluate_case(
    token: str,
    case: EvalCase,
) -> EvalResult:
    """
    执行并评估单条用例。
    """

    status_code, response_json, latency_ms = call_chat(
        token=token,
        question=case.question,
        metadata_filter=case.metadata_filter,
    )

    if case.case_type == "qa":
        return evaluate_qa_case(case, status_code, response_json, latency_ms)

    if case.case_type == "no_answer":
        return evaluate_no_answer_case(case, status_code, response_json, latency_ms)

    if case.case_type == "permission":
        return evaluate_permission_case(case, status_code, response_json, latency_ms)

    raise ValueError(f"未知 case_type：{case.case_type}")


def average(values: List[float]) -> Optional[float]:
    """
    安全计算平均值。
    """

    if not values:
        return None

    return sum(values) / len(values)


def percentile(values: List[float], p: float) -> Optional[float]:
    """
    计算百分位数。

    p 示例：
        50 表示 P50
        95 表示 P95
    """

    if not values:
        return None

    sorted_values = sorted(values)

    index = int(round((p / 100) * (len(sorted_values) - 1)))

    return sorted_values[index]



def summarize_results(results: List[EvalResult]) -> Dict[str, Any]:
    """
    汇总评估结果。
    """

    qa_results = [item for item in results if item.case_type == "qa"]
    no_answer_results = [item for item in results if item.case_type == "no_answer"]
    permission_results = [item for item in results if item.case_type == "permission"]

    latency_values = [item.latency_ms for item in results]

    faithfulness_values = [
        item.faithfulness
        for item in qa_results
        if item.faithfulness is not None
    ]

    summary = {
        "total_cases": len(results),
        "passed_cases": sum(1 for item in results if item.passed),
        "pass_rate": sum(1 for item in results if item.passed) / len(results) if results else 0,

        "qa_cases": len(qa_results),
        "hit_at_1": average([item.hit_at_1 for item in qa_results if item.hit_at_1 is not None]),
        "hit_at_3": average([item.hit_at_3 for item in qa_results if item.hit_at_3 is not None]),
        "hit_at_5": average([item.hit_at_5 for item in qa_results if item.hit_at_5 is not None]),
        "precision_at_5": average([item.precision_at_5 for item in qa_results if item.precision_at_5 is not None]),
        "mrr": average([item.mrr for item in qa_results if item.mrr is not None]),
        "fact_coverage": average([item.fact_coverage for item in qa_results if item.fact_coverage is not None]),

        # 如果 API 没有返回 chunk 内容，这个可能是 None
        "faithfulness": average(faithfulness_values),

        "no_answer_cases": len(no_answer_results),
        "no_answer_accuracy": average([
            item.no_answer_correct
            for item in no_answer_results
            if item.no_answer_correct is not None
        ]),

        "permission_cases": len(permission_results),
        "permission_accuracy": average([
            item.permission_correct
            for item in permission_results
            if item.permission_correct is not None
        ]),

        "latency": {
            "avg_ms": average(latency_values),
            "p50_ms": percentile(latency_values, 50),
            "p95_ms": percentile(latency_values, 95),
            "min_ms": min(latency_values) if latency_values else None,
            "max_ms": max(latency_values) if latency_values else None,
        },
    }

    return summary


def format_percent(value: Optional[float]) -> str:
    """
    格式化百分比。
    """

    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def format_number(value: Optional[float]) -> str:
    """
    格式化数字。
    """

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def format_ms(value: Optional[float]) -> str:
    """
    格式化毫秒。
    """

    if value is None:
        return "N/A"

    return f"{value:.2f} ms"


def write_json_report(
    results: List[EvalResult],
    summary: Dict[str, Any],
) -> None:
    """
    输出 JSON 报告。
    """

    ensure_report_dir()

    report = {
        "api_base_url": API_BASE_URL,
        "summary": summary,
        "results": [asdict(item) for item in results],
    }

    with REPORT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def write_markdown_report(
    results: List[EvalResult],
    summary: Dict[str, Any],
) -> None:
    """
    输出 Markdown 报告。
    """

    ensure_report_dir()

    lines: List[str] = []

    lines.append("# Enterprise-RAG 最终评估报告")
    lines.append("")
    lines.append("## 1. 总览")
    lines.append("")
    lines.append(f"- API 地址：`{API_BASE_URL}`")
    lines.append(f"- 总用例数：{summary['total_cases']}")
    lines.append(f"- 通过用例数：{summary['passed_cases']}")
    lines.append(f"- 总通过率：{format_percent(summary['pass_rate'])}")
    lines.append("")

    lines.append("## 2. 检索与排序指标")
    lines.append("")
    lines.append("| 指标 | 结果 |")
    lines.append("| --- | ---: |")
    lines.append(f"| Hit@1 | {format_percent(summary['hit_at_1'])} |")
    lines.append(f"| Hit@3 | {format_percent(summary['hit_at_3'])} |")
    lines.append(f"| Hit@5 | {format_percent(summary['hit_at_5'])} |")
    lines.append(f"| Precision@5 | {format_percent(summary['precision_at_5'])} |")
    lines.append(f"| MRR | {format_number(summary['mrr'])} |")
    lines.append("")

    lines.append("## 3. 答案质量指标")
    lines.append("")
    lines.append("| 指标 | 结果 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 关键事实覆盖率 | {format_percent(summary['fact_coverage'])} |")
    lines.append(f"| 忠实度 Faithfulness | {format_percent(summary['faithfulness'])} |")
    lines.append(f"| 无答案拒答准确率 | {format_percent(summary['no_answer_accuracy'])} |")
    lines.append(f"| 权限拦截准确率 | {format_percent(summary['permission_accuracy'])} |")
    lines.append("")
    lines.append("> 注：如果 API 的 sources 未返回 chunk 正文，忠实度会显示为 N/A。")
    lines.append("")

    lines.append("## 4. 响应时间")
    lines.append("")
    latency = summary["latency"]
    lines.append("| 指标 | 结果 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 平均响应时间 | {format_ms(latency['avg_ms'])} |")
    lines.append(f"| P50 | {format_ms(latency['p50_ms'])} |")
    lines.append(f"| P95 | {format_ms(latency['p95_ms'])} |")
    lines.append(f"| 最快请求 | {format_ms(latency['min_ms'])} |")
    lines.append(f"| 最慢请求 | {format_ms(latency['max_ms'])} |")
    lines.append("")

    lines.append("## 5. 明细")
    lines.append("")
    lines.append("| Case ID | 类型 | 通过 | 耗时 | Hit@5 | Fact Coverage | 问题 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")

    for item in results:
        lines.append(
            f"| `{item.case_id}` "
            f"| {item.case_type} "
            f"| {'✅' if item.passed else '❌'} "
            f"| {format_ms(item.latency_ms)} "
            f"| {format_percent(item.hit_at_5)} "
            f"| {format_percent(item.fact_coverage)} "
            f"| {item.question} |"
        )

    lines.append("")

    failed_results = [item for item in results if not item.passed]

    if failed_results:
        lines.append("## 6. 失败用例")
        lines.append("")

        for item in failed_results:
            lines.append(f"### {item.case_id}")
            lines.append("")
            lines.append(f"- 问题：{item.question}")
            lines.append(f"- 错误：{item.error}")
            lines.append(f"- 状态码：{item.status_code}")
            lines.append(f"- Sources：{item.sources}")
            lines.append(f"- Answer：{item.answer}")
            lines.append("")

    with REPORT_MD_PATH.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def print_summary(summary: Dict[str, Any]) -> None:
    """
    在控制台打印核心指标。
    """

    print("\n" + "=" * 100)
    print("Enterprise-RAG 最终评估结果")
    print("=" * 100)

    print(f"总通过率: {format_percent(summary['pass_rate'])}")
    print(f"Hit@1: {format_percent(summary['hit_at_1'])}")
    print(f"Hit@3: {format_percent(summary['hit_at_3'])}")
    print(f"Hit@5: {format_percent(summary['hit_at_5'])}")
    print(f"Precision@5: {format_percent(summary['precision_at_5'])}")
    print(f"MRR: {format_number(summary['mrr'])}")
    print(f"关键事实覆盖率: {format_percent(summary['fact_coverage'])}")
    print(f"忠实度 Faithfulness: {format_percent(summary['faithfulness'])}")
    print(f"无答案拒答准确率: {format_percent(summary['no_answer_accuracy'])}")
    print(f"权限拦截准确率: {format_percent(summary['permission_accuracy'])}")

    latency = summary["latency"]
    print(f"平均响应时间: {format_ms(latency['avg_ms'])}")
    print(f"P50: {format_ms(latency['p50_ms'])}")
    print(f"P95: {format_ms(latency['p95_ms'])}")

    print("=" * 100)
    print(f"JSON 报告: {REPORT_JSON_PATH}")
    print(f"Markdown 报告: {REPORT_MD_PATH}")
    print("=" * 100)


def run_final_eval() -> None:
    """
    运行最终评估。
    """

    print("=" * 100)
    print("开始运行 Enterprise-RAG 最终 API 评估")
    print(f"API_BASE_URL: {API_BASE_URL}")
    print("=" * 100)

    check_health()

    token = login()

    cases = build_eval_cases()
    results: List[EvalResult] = []

    print_step(f"开始执行评估用例，共 {len(cases)} 条")

    for case in cases:
        print(f"\n[EVAL] {case.case_id} - {case.question}")

        try:
            result = evaluate_case(token, case)
        except Exception as exc:
            result = EvalResult(
                case_id=case.case_id,
                question=case.question,
                case_type=case.case_type,

                status_code=0,
                latency_ms=0,

                answer="",
                sources=[],

                hit_at_1=None,
                hit_at_3=None,
                hit_at_5=None,
                precision_at_5=None,
                mrr=None,

                fact_coverage=None,
                faithfulness=None,

                no_answer_correct=None,
                permission_correct=None,

                passed=False,
                error=str(exc),
            )

        results.append(result)

        if result.passed:
            print(f"[PASS] {case.case_id} latency={result.latency_ms:.2f}ms")
        else:
            print(f"[FAIL] {case.case_id} error={result.error}")

    summary = summarize_results(results)

    write_json_report(results, summary)
    write_markdown_report(results, summary)

    print_summary(summary)


if __name__ == "__main__":
    run_final_eval()