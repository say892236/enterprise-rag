# eval/questions.py

"""
统一评估问题集。

说明：
1. EVAL_CASES 保存问题及其标准标签。
2. TEST_QUESTIONS 保留原来的字符串列表，兼容现有评估脚本。
3. 使用 source + header_2 判断检索结果是否相关。
4. 不使用 chunk_id 作为标准标签，因为数据库重建后 chunk_id 可能变化。
"""


EVAL_CASES = [
    {
        "id": "travel_hotel_standard",
        "question": "员工出差住宿标准是什么？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "travel_policy.md",
                "header_2": "3. 住宿标准",
            },
            {
                "source": "travel_policy.md",
                "header_2": "8. FAQ",
            },
        ],
    },
    {
        "id": "shanghai_hotel_limit",
        "question": "上海出差普通员工住宿上限是多少？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "travel_policy.md",
                "header_2": "3. 住宿标准",
            },
            {
                "source": "travel_policy.md",
                "header_2": "8. FAQ",
            },
        ],
    },
    {
        "id": "travel_reimbursement_deadline",
        "question": "员工出差结束后多久内必须提交报销？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "travel_policy.md",
                "header_2": "7. 报销要求",
            },
            {
                "source": "travel_policy.md",
                "header_2": "8. FAQ",
            },
        ],
    },
    {
        "id": "flight_condition",
        "question": "高铁几个小时以上可以坐飞机？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "travel_policy.md",
                "header_2": "4. 交通标准",
            },
            {
                "source": "travel_policy.md",
                "header_2": "8. FAQ",
            },
        ],
    },
    {
        "id": "city_taxi_limit",
        "question": "单次市内打车超过200元需要怎么处理？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "travel_policy.md",
                "header_2": "4. 交通标准",
            },
            {
                "source": "travel_policy.md",
                "header_2": "8. FAQ",
            },
        ],
    },
    {
        "id": "colloquial_shanghai_hotel_limit",
        "question": "我去上海出差，普通员工酒店最多能报多少？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "travel_policy.md",
                "header_2": "3. 住宿标准",
            },
            {
                "source": "travel_policy.md",
                "header_2": "8. FAQ",
            },
        ],
    },
    {
        "id": "colloquial_hotel_standard",
        "question": "住宿能报多少？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "travel_policy.md",
                "header_2": "3. 住宿标准",
            },
            {
                "source": "travel_policy.md",
                "header_2": "8. FAQ",
            },
        ],
    },
    {
        "id": "generic_reimbursement_deadline",
        "question": "报销多久内提交？",
        "case_type": "qa",
        "expected_targets": [
            {
                "source": "reimbursement_policy.md",
                "header_2": "4. 审批流程",
            },
            {
                "source": "reimbursement_policy.md",
                "header_2": "7. FAQ",
            },
        ],
    }
]


# 兼容 baseline_eval.py、reranker_eval.py 等现有脚本。
# 强制声明结果是字符串列表。
TEST_QUESTIONS: list[str] = [
    str(case["question"])
    for case in EVAL_CASES
]


# 通过问题快速获取对应的标准标签。
# str() 明确告诉类型检查器：字典键一定是字符串。
EVAL_CASE_MAP: dict[str, dict] = {
    str(case["question"]): case
    for case in EVAL_CASES
}


