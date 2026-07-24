# eval/negative_answer_eval.py

"""
制度否定类问题评估脚本。

作用：
1. 测试知识库中有明确规则或上位规则的问题。
2. 这类问题不应该拒答。
3. 系统应该根据制度回答“不可以 / 不允许 / 不予报销 / 不符合要求”等否定结论。
4. 同时应该返回 sources，证明答案有知识库依据。

运行方式：
python -m eval.negative_answer_eval
"""

from pipelines.rag_pipeline import build_hybrid_rag_chain


NO_INFO_ANSWER = "知识库中未找到相关信息。"

NEGATIVE_ANSWER_TEST_CASES = [
    {
        "question": "员工可以报销奶茶吗？",
        "expected_source": "reimbursement_policy.md",
        "required_keywords": ["奶茶", "报销"],
        "negative_keywords": ["不可以", "不能", "不得", "不可报销", "不予报销", "不可"],
    },
    {
        "question": "交通罚款可以报销吗？",
        "expected_source": "reimbursement_policy.md",
        "required_keywords": ["罚款", "报销"],
        "negative_keywords": ["不可以", "不能", "不得", "不可报销", "不予报销", "不可"],
    },
    {
        "question": "私人聚餐可以报销吗？",
        "expected_source": "reimbursement_policy.md",
        "required_keywords": ["私人", "报销"],
        "negative_keywords": ["不可以", "不能", "不得", "不可报销", "不予报销", "不可"],
    },
    {
        "question": "电子发票截图可以作为报销凭证吗？",
        "expected_source": "reimbursement_policy.md",
        "required_keywords": ["截图", "发票"],
        "negative_keywords": ["不可以", "不能", "不得", "不可", "不得仅上传截图"],
    },
    {
        "question": "员工可以虚假报销吗？",
        "expected_source": "reimbursement_policy.md",
        "required_keywords": ["虚假报销"],
        "negative_keywords": ["不可以", "不能", "不得", "不可", "严重违纪", "取消报销", "追回款项"],
    },
    {
        "question": "出差可以随便住五星级酒店吗？",
        "expected_source": "travel_policy.md",
        "required_keywords": ["住"],
        "negative_keywords": ["不可以", "不能", "不得", "不可", "标准", "审批"],
    },
]


def print_separator() -> None:
    """
    打印分隔线。
    """

    print("=" * 100)


def contains_required_keywords(
    answer: str,
    required_keywords: list[str],
) -> bool:
    """
    判断答案中是否包含所有必要业务关键词。

    例如：
    问题：交通罚款可以报销吗？

    答案中至少应该包含：
    - 罚款
    - 报销
    """

    return all(
        keyword in answer
        for keyword in required_keywords
    )


def contains_negative_keyword(
    answer: str,
    negative_keywords: list[str],
) -> bool:
    """
    判断答案中是否包含任意一个否定表达。

    LLM 可能会说：
    - 不可以报销
    - 不能报销
    - 不得报销
    - 不予报销
    - 属于不可报销费用

    这些都应该视为正确的否定回答。
    """

    return any(
        keyword in answer
        for keyword in negative_keywords
    )

def has_expected_source(
    sources: list[dict],
    expected_source: str,
) -> bool:
    """
    判断 sources 中是否包含预期来源文档。
    """

    returned_sources = {
        source.get("source")
        for source in sources
    }

    return expected_source in returned_sources

def evaluate_case(
    rag_chain,
    test_case: dict,
) -> bool:
    """
    评估单个制度否定类问题。

    通过标准：
    1. answer 不能是固定拒答文本。
    2. sources 不能为空。
    3. sources 中必须包含 expected_source。
    4. answer 中必须包含必要业务关键词。
    5. answer 中至少包含一个否定表达。
    """

    question = test_case["question"]
    expected_source = test_case["expected_source"]

    # 新字段：必须命中的业务关键词
    required_keywords = test_case["required_keywords"]

    # 新字段：否定表达，只要命中任意一个即可
    negative_keywords = test_case["negative_keywords"]

    result = rag_chain.invoke(
        question=question,
        retrieve_top_k=8,
        rerank_top_k=3,
        metadata_filter=None,
        enable_rewrite=True,
        enable_rerank=True,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    rewritten_question = result.get("rewritten_question")
    confidence = result.get("confidence")

    # 1. 这类问题不应该触发拒答
    not_refused = answer != NO_INFO_ANSWER

    # 2. 应该有引用来源
    has_sources = len(sources) > 0

    # 3. sources 中应该包含预期来源文档
    source_passed = has_expected_source(
        sources=sources,
        expected_source=expected_source,
    )

    # 4. 答案中应该包含必要业务关键词
    required_keywords_passed = contains_required_keywords(
        answer=answer,
        required_keywords=required_keywords,
    )

    # 5. 答案中应该包含任意一个否定表达
    negative_keyword_passed = contains_negative_keyword(
        answer=answer,
        negative_keywords=negative_keywords,
    )

    passed = (
        not_refused
        and has_sources
        and source_passed
        and required_keywords_passed
        and negative_keyword_passed
    )

    print_separator()
    print(f"问题：{question}")
    print(f"改写后：{rewritten_question}")
    print(f"回答：{answer}")
    print(f"预期来源：{expected_source}")
    print(f"sources 数量：{len(sources)}")
    print(f"confidence：{confidence}")

    if sources:
        print("实际来源：")
        for index, source in enumerate(sources, start=1):
            print(
                f"  {index}. "
                f"{source.get('source')} | "
                f"{source.get('header_2')} | "
                f"rerank_score={source.get('rerank_score')}"
            )

    if passed:
        print("评估结果：[PASS] 正确给出制度否定回答")
    else:
        print("评估结果：[FAIL] 制度否定回答不符合预期")

        if not not_refused:
            print("原因：该问题有制度依据，但系统错误拒答。")

        if not has_sources:
            print("原因：sources 为空，缺少知识库依据。")

        if not source_passed:
            print(f"原因：sources 中没有预期来源文档 {expected_source}")

        if not required_keywords_passed:
            print(f"原因：answer 中缺少必要业务关键词 {required_keywords}")

        if not negative_keyword_passed:
            print(f"原因：answer 中缺少否定表达，预期任意命中：{negative_keywords}")

    return passed

def main() -> None:
    """
    执行制度否定类问题评估。
    """

    print_separator()
    print("开始制度否定类问题评估")
    print_separator()

    rag_chain = build_hybrid_rag_chain()

    total_count = len(NEGATIVE_ANSWER_TEST_CASES)
    pass_count = 0

    for test_case in NEGATIVE_ANSWER_TEST_CASES:
        passed = evaluate_case(
            rag_chain=rag_chain,
            test_case=test_case,
        )

        if passed:
            pass_count += 1

    fail_count = total_count - pass_count

    print_separator()
    print("制度否定类问题评估完成")
    print_separator()
    print(f"总问题数：{total_count}")
    print(f"通过数量：{pass_count}")
    print(f"失败数量：{fail_count}")
    print(f"通过率：{pass_count / total_count:.2%}")

    if fail_count > 0:
        raise AssertionError("存在制度否定类问题未正确回答，请检查检索、重排或 Prompt。")


if __name__ == "__main__":
    main()