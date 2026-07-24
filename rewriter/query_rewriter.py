# rewriter/query_rewriter.py

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


class QueryRewriter:
    """
    企业级 RAG 查询改写器。

    作用：
    - 将用户的口语化、模糊问题改写成更适合知识库检索的问题
    - 不回答问题
    - 不编造实体、金额、制度
    - 如果问题已经清楚，则尽量保持原问题
    """

    def __init__(
        self,
        llm_model: str = "deepseek-v4-flash",
        llm_provider: str = "deepseek",
        temperature: float = 0,
    ):
        self.llm = init_chat_model(
            model=llm_model,
            model_provider=llm_provider,
            temperature=temperature,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
        你是企业知识库 RAG 系统中的查询改写器。

        你的任务不是回答问题，而是把用户问题改写成更适合检索企业制度文档的问题。

        规则：
        1. 只输出改写后的问题，不要输出解释。
        2. 不要回答用户问题。
        3. 不要编造制度、金额、日期、审批人或公司信息。
        4. 如果用户问题已经清楚，可以保持原问题。
        5. 如果用户问题过于口语化，需要改写得更正式、更完整。
        6. 如果用户问题缺少上下文，但仍能判断大致领域，可以补全为常见企业制度问法。
        7. 如果用户问题完全无法判断含义，则保持原问题，不要强行补全。
        8. 改写后的问题应适合用于向量检索和关键词检索。
        9. 如果用户问题包含明确的制度编号、文档编号、条款编号，例如 FIN-TR-2026-01、FAQ-003、3.2、5.1，应尽量保留编号原样。
        10. 不要改变用户问题的查询意图。例如用户问“是什么制度”，不要改写成“具体内容是什么”；用户问“编号是什么”，不要改写成“制度内容是什么”。
        11. 对编号类查询，优先保持原问题或只做轻微补全。

        示例：

        用户问题：住宿能报多少？
        改写：员工出差住宿报销标准是多少？

        用户问题：报销多久内提交？
        改写：员工报销申请需要在多久内提交？

        用户问题：高铁太久了能不能坐飞机？
        改写：员工出差时高铁时间超过多久可以选择飞机？

        用户问题：FIN-TR-2026-01 是什么制度？
        改写：FIN-TR-2026-01 是什么制度？

        用户问题：FAQ-003 是什么？
        改写：FAQ-003 是什么？

        用户问题：这个能报吗？
        改写：这个能报吗？
        """,
                ),
                (
                    "human",
                    "用户问题：{question}\n改写后的检索问题：",
                ),
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        """
        改写用户问题。
        """

        rewritten_question = self.chain.invoke(
            {
                "question": question,
            }
        )

        return rewritten_question.strip()