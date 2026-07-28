
# tests/test_confidence_guard.py

"""
低置信度拒答机制单元测试。

测试目标：
1. 不依赖 FastAPI 服务。
2. 不调用 LLM。
3. 不调用向量库。
4. 只测试 chains/confidence_guard.py 里的规则逻辑。
"""

import sys

from langchain_core.documents import Document

from chains.confidence_guard import RetrievalConfidenceGuard


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def test_empty_documents_should_fail() -> None:
    """
    测试没有检索到任何文档时，应该判定为低置信度。
    """

    guard = RetrievalConfidenceGuard(
        min_overlap_count=1,
        check_top_n=3,
    )

    result = guard.check(
        query="员工出差住宿标准是什么？",
        documents=[],
    )

    assert result.passed is False
    assert result.reason == "没有检索到任何相关文档"
    assert result.overlap_count == 0

    print_pass("没有检索结果时触发低置信度拒答")


def test_empty_query_tokens_should_fail() -> None:
    """
    测试用户问题缺少有效关键词时，应该判定为低置信度。

    例如只问：
    - 什么？
    - 这个？
    - 有没有？
    这些问题本身没有明确业务关键词。
    """

    guard = RetrievalConfidenceGuard(
        min_overlap_count=1,
        check_top_n=3,
    )

    docs = [
        Document(
            page_content="员工出差期间应遵守差旅制度，并按照规定提交报销申请。",
            metadata={
                "source": "travel_policy.md",
            },
        )
    ]

    result = guard.check(
        query="什么？",
        documents=docs,
    )

    print(result.to_dict())

    assert result.passed is False
    assert result.reason == "用户问题缺少有效关键词"
    assert result.overlap_count == 0

    print_pass("问题缺少有效关键词时触发低置信度拒答")


def test_irrelevant_canteen_question_should_fail() -> None:
    """
    测试知识库无关问题应该被拦截。

    问题：
    公司食堂几点开门？

    检索结果：
    员工手册中的考勤管理 / 工作时间。

    虽然文档里有“工作时间”，但没有“食堂”。
    因此不能认为这是有效命中。
    """

    guard = RetrievalConfidenceGuard(
        min_overlap_count=1,
        check_top_n=3,
    )

    docs = [
        Document(
            page_content=(
                "## 5. 考勤管理\n"
                "公司实行标准工时制。工作日为周一至周五，每天工作 8 小时。\n"
                "默认工作时间为上午 09:30 至 12:00，下午 13:30 至 18:30。"
            ),
            metadata={
                "source": "employee_handbook.md",
                "header_2": "5. 考勤管理",
            },
        ),
        Document(
            page_content=(
                "员工申请远程办公需提前 1 个工作日提交申请。"
            ),
            metadata={
                "source": "员工手册.md",
                "header_2": "5. 考勤管理",
            },
        ),
    ]

    result = guard.check(
        query="公司附近有没有健身房？",
        documents=docs,
    )

    print(result.to_dict())


    assert result.passed is False
    assert "问题关键词与检索结果缺少有效重叠" in result.reason
    assert result.overlap_count == 0
    assert "健身房" in result.query_tokens
    assert result.matched_tokens == []

    print_pass("有没有健身房这类无关问题触发低置信度拒答")


def test_relevant_travel_hotel_question_should_pass() -> None:
    """
    测试相关问题应该通过置信度检查。

    问题：
    员工出差住宿标准是什么？

    文档：
    差旅制度中的住宿标准。

    问题关键词和文档正文存在有效重叠，应该允许进入 LLM。
    """

    guard = RetrievalConfidenceGuard(
        min_overlap_count=1,
        check_top_n=3,
    )

    docs = [
        Document(
            page_content=(
                "## 3. 住宿标准\n"
                "员工出差期间应按照城市等级和职级执行住宿标准。"
                "一类城市普通员工住宿标准为 600 元/晚。"
            ),
            metadata={
                "source": "差旅住宿与交通标准.md",
                "header_2": "3. 住宿标准",
            },
        )
    ]

    result = guard.check(
        query="员工出差住宿标准是什么？",
        documents=docs,
    )

    print(repr(result.matched_tokens))

    print(
        "出差" in result.matched_tokens,
        "住宿" in result.matched_tokens
    )

    assert result.passed is True
    assert result.reason == "检索结果通过关键词重叠检查"
    assert result.overlap_count >= 1

    # 至少应该命中这些关键词中的一个。
    assert any(
        token in result.matched_tokens
        for token in ["出差", "住宿"]
    )


    print_pass("相关差旅住宿问题通过置信度检查")


def test_to_dict_should_return_serializable_data() -> None:
    """
    测试 ConfidenceResult.to_dict() 返回普通 dict。

    这个结果后续可以安全写入日志或 RAG 返回结果。
    """

    guard = RetrievalConfidenceGuard(
        min_overlap_count=1,
        check_top_n=3,
    )

    docs = [
        Document(
            page_content="员工出差住宿标准按照城市等级执行。",
            metadata={
                "source": "差旅住宿与交通标准.md",
            },
        )
    ]

    result = guard.check(
        query="出差住宿标准是什么？",
        documents=docs,
    )

    data = result.to_dict()

    assert isinstance(data, dict)
    assert "passed" in data
    assert "reason" in data
    assert "query_tokens" in data
    assert "matched_tokens" in data
    assert "overlap_count" in data

    print_pass("ConfidenceResult 可以转换为普通 dict")


def run_all_tests() -> None:
    """
    按顺序运行低置信度拒答机制测试。
    """

    print("=" * 100)
    print("开始运行低置信度拒答机制单元测试")
    print("=" * 100)

    test_empty_documents_should_fail()
    test_empty_query_tokens_should_fail()
    test_irrelevant_canteen_question_should_fail()
    test_relevant_travel_hotel_question_should_pass()
    test_to_dict_should_return_serializable_data()

    print("=" * 100)
    print("低置信度拒答机制单元测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)