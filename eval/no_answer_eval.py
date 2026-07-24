# eval/no_answer_eval.py

"""
低置信度拒答评估脚本。

作用：
1. 专门测试知识库不存在的问题。
2. 检查 RAG 是否返回固定拒答文本。
3. 检查拒答时 sources 是否为空。
4. 用于评估低置信度拒答机制是否稳定。

运行方式：
python -m eval.no_answer_eval
"""

from pipelines.rag_pipeline import build_hybrid_rag_chain


# 固定拒答文本，必须和 chains/rag_chain.py 中保持一致
NO_INFO_ANSWER = "知识库中未找到相关信息。"


# 知识库中不存在的问题集合
NO_ANSWER_QUESTIONS = [
    "公司食堂几点开门？",
    "老板今年几岁？",
    "公司附近有没有停车场？",
    "公司有没有健身房？",
    "公司班车几点发车？",
    "公司年会在哪里举办？",
    "公司有没有宠物友好政策？",
]


def print_separator() -> None:
    """
    打印分隔线，方便查看输出。
    """

    print("=" * 100)


def evaluate_question(
    rag_chain,
    question: str,
) -> bool:
    """
    评估单个无答案问题。

    评估标准：
    1. answer 必须等于固定拒答文本。
    2. sources 必须为空列表。

    返回：
        True：通过
        False：失败
    """

    result = rag_chain.invoke(
        question=question,
        retrieve_top_k=6,
        rerank_top_k=3,
        metadata_filter=None,
        enable_rewrite=True,
        enable_rerank=True,
    )

    answer = result.get("answer")
    sources = result.get("sources", [])
    rewritten_question = result.get("rewritten_question")
    confidence = result.get("confidence")

    answer_passed = answer == NO_INFO_ANSWER
    sources_passed = sources == []

    passed = answer_passed and sources_passed

    print_separator()
    print(f"问题：{question}")
    print(f"改写后：{rewritten_question}")
    print(f"回答：{answer}")
    print(f"sources 数量：{len(sources)}")
    print(f"confidence：{confidence}")

    if passed:
        print("评估结果：[PASS] 正确拒答")
    else:
        print("评估结果：[FAIL] 拒答不符合预期")

        if not answer_passed:
            print(f"原因：answer 不等于固定拒答文本，实际 answer={answer}")

        if not sources_passed:
            print(f"原因：拒答时 sources 不为空，实际 sources={sources}")

    return passed


def main() -> None:
    """
    执行低置信度拒答评估。
    """

    print_separator()
    print("开始低置信度拒答评估")
    print_separator()

    rag_chain = build_hybrid_rag_chain()

    total_count = len(NO_ANSWER_QUESTIONS)
    pass_count = 0

    for question in NO_ANSWER_QUESTIONS:
        passed = evaluate_question(
            rag_chain=rag_chain,
            question=question,
        )

        if passed:
            pass_count += 1

    fail_count = total_count - pass_count

    print_separator()
    print("低置信度拒答评估完成")
    print_separator()
    print(f"总问题数：{total_count}")
    print(f"通过数量：{pass_count}")
    print(f"失败数量：{fail_count}")
    print(f"通过率：{pass_count / total_count:.2%}")

    if fail_count > 0:
        raise AssertionError("存在未正确拒答的问题，请检查 confidence_guard 或 Prompt 规则。")


if __name__ == "__main__":
    main()