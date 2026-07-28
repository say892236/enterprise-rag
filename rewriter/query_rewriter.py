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
            
                    你的任务：
                    根据【历史对话】和【当前问题】，
                    生成适合知识库检索的问题。
            
                    规则：
            
                    1. 只输出改写后的问题。
                    2. 不回答问题。
                    3. 不编造信息。
                    4. 如果当前问题依赖历史上下文，需要补全缺失信息。
                    5. 保留制度编号、金额、日期等关键实体。
                    6. 如果当前问题已经完整，不要修改。
            
            
                    示例：
            
                    历史：
                    用户：上海出差住宿标准是多少？
                    助手：上海属于一类城市，普通员工600元。
            
                    当前：
                    普通员工呢？
            
                    输出：
                    上海出差住宿标准中普通员工住宿标准是多少？
            
                    """
                ),
                (
                    "human",
                    """
                    历史对话：
                    {chat_history}
            
                    当前问题：
                    {question}
            
                    改写后的检索问题：
                    """
                )
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str,chat_history=None,) -> str:
        """
           改写用户问题。

           参数：
           question:
               当前用户问题

           chat_history:
               Redis 中保存的历史对话

           作用：
           将多轮对话中的省略问题，
           转换成独立完整的问题。
           """

        history_text = self.format_chat_history(chat_history)

        rewritten_question = self.chain.invoke(
            {
                "question": question,
                "chat_history": chat_history or []
            }
        )


        return rewritten_question.strip()

    def format_chat_history(self, chat_history):
        """
        将历史消息转换为 Query Rewrite 可理解的文本。

        输入:

        [
            {
                "role": "user",
                "content": "上海住宿标准是多少?"
            },
            {
                "role": "assistant",
                "content": "普通员工600元/晚"
            }
        ]


        输出:

        用户:
        上海住宿标准是多少?

        AI:
        普通员工600元/晚
        """

        if not chat_history:
            return "暂无历史对话"

        messages = []

        for item in chat_history:

            role = item.get("role")

            content = item.get("content")

            if role == "user":
                messages.append(
                    f"用户：{content}"
                )

            elif role == "assistant":
                messages.append(
                    f"AI：{content}"
                )

        return "\n".join(messages)