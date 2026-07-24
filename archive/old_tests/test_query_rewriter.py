# test_query_rewriter.py

from rewriter.query_rewriter import QueryRewriter


def main():
    rewriter = QueryRewriter(
        llm_model="deepseek-v4-flash",
        llm_provider="deepseek",
        temperature=0,
    )

    questions = [
        "住宿能报多少？",
        "报销多久内提交？",
        "高铁太久了能不能坐飞机？",
        "我去上海出差，普通员工酒店最多能报多少？",
        "员工买咖啡可以报销吗？",
        "出差可以住五星级酒店吗？",
        "公司食堂几点开门？",
        "这个能报吗？",
        "FIN-TR-2026-01 是什么制度？",
        "FAQ-003 是什么？",
    ]

    for question in questions:
        rewritten = rewriter.rewrite(question)

        print("=" * 80)
        print("原始问题：")
        print(question)
        print("改写后：")
        print(rewritten)


if __name__ == "__main__":
    main()