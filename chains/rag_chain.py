# chains/rag_chain.py

from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from reranker.bge_reranker import BGEReranker
from retriever.chroma_retriever import ChromaRetriever
from rewriter.query_rewriter import QueryRewriter

load_dotenv()


class EnterpriseRAGChain:
    """
    企业级 RAG 基础链路。

    当前版本：
    - ChromaRetriever 检索
    - 拼接 context
    - Prompt 约束回答
    - LLM 生成答案
    - 返回答案和引用来源

    暂不包含：
    - Query Rewrite
    - Reranker
    - Hybrid Search
    - Context Compression
    """

    def __init__(
        self,
        retriever: ChromaRetriever,
        reranker: Optional[BGEReranker] = None,
        rewriter: Optional[QueryRewriter]=None,
        llm_model: str = "deepseek-v4-flash",
        llm_provider: str = "deepseek",
        temperature: float = 0.0,
        confidence_guard: Optional[BGEReranker] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.rewriter = rewriter
        self.llm = init_chat_model(
            model=llm_model,
            model_provider=llm_provider,
            temperature=temperature,
        )
        self.confidence_guard = confidence_guard

        parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
你是企业知识库问答助手。

你必须严格根据给定的【知识库上下文】回答问题。

核心规则：
1. 只能使用【知识库上下文】中的信息回答。
2. 不得使用常识、经验、猜测或外部知识补充答案。
3. 如果知识库中完全没有相关信息，回答：知识库中未找到相关信息。
4. 如果知识库没有直接提到用户问题中的具体对象，但上下文中存在可适用的上位制度、金额标准、审批规则或限制条件，不要直接拒答。
5. 如果用户问题可以通过上下文中的标准表、金额区间、条件规则回答，但还缺少城市、职级、时间、部门等必要条件，
   不要说“知识库未直接说明该问题”；应先给出可参考的标准或规则，再说明需要用户补充哪些条件才能确定唯一答案。
6. 如果知识库中只有相关上位规则，但无法直接推出明确结论，才说明“知识库未直接说明该问题”，再列出可参考规则。
7. 不要使用“理论上可以”“应该可以”“大概率”“一般来说”等推测表达。
8. 不要编造制度、金额、日期、流程、审批人或例外情况。
9. 涉及金额、天数、审批规则时，必须严格引用上下文中的原始规定。
10. 回答要简洁，优先回答用户直接问题，不要过度展开。
11. 回答末尾不要编造来源，来源由系统单独提供。

回答格式要求：
- 如果有明确答案：直接回答，并列出关键规则。
- 如果没有直接答案，但有相关制度：先回答“知识库未直接说明该问题”，再列出可参考规则。
- 如果完全没有相关信息：只回答“知识库中未找到相关信息。”
- 如果有标准表或条件规则：先列出标准或规则，再说明还需要哪些条件才能确定唯一答案。
- 如果只有相关制度但不能直接推出结论：先回答“知识库未直接说明该问题”，再列出可参考规则。


【知识库上下文】
{context}
""",
                ),
                ("human", "{question}"),
            ]
        )

        self.chain = self.prompt | self.llm | parser

    def format_docs(self, docs: List[Document]) -> str:
        """
        将检索到的 Document 列表格式化成 Prompt 里的 context。
        """

        context_parts = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            header_1 = doc.metadata.get("header_1", "")
            header_2 = doc.metadata.get("header_2", "")
            chunk_id = doc.metadata.get("chunk_id", "")
            score = doc.metadata.get("score", "")

            context = f"""
【文档片段 {i}】
source: {source}
header_1: {header_1}
header_2: {header_2}
chunk_id: {chunk_id}
score: {score}

内容：
{doc.page_content}
"""
            context_parts.append(context)

        return "\n".join(context_parts)

    def format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        格式化引用来源。

        作用：
        1. 从 Document 中提取必要 metadata。
        2. 保留检索、重排相关分数。
        3. 加入 page_content，供 API 层生成 content_preview。
        """

        sources = []

        for doc in documents:
            metadata = doc.metadata

            source = {
                # ===== 基础溯源信息 =====
                "source": metadata.get("source"),
                "file_path": metadata.get("file_path"),
                "file_type": metadata.get("file_type"),
                "header_1": metadata.get("header_1"),
                "header_2": metadata.get("header_2"),
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),

                # ===== 文档正文 =====
                # page_content：保留 LangChain Document 的原始命名
                # content：给 API / 前端 / 评估脚本使用的统一字段
                # content_preview：前端展示用摘要
                "page_content": doc.page_content,
                "content": doc.page_content,
                "content_preview": doc.page_content[:200],

                # Dense 检索分数
                "score": metadata.get("score"),
                "chroma_score": metadata.get("chroma_score"),
                "milvus_score": metadata.get("milvus_score"),
                "dense_rank": metadata.get("dense_rank"),
                "dense_rrf_score": metadata.get("dense_rrf_score"),

                # BM25 检索分数
                "bm25_score": metadata.get("bm25_score"),
                "bm25_rank": metadata.get("bm25_rank"),
                "bm25_rrf_score": metadata.get("bm25_rrf_score"),

                # Hybrid / Reranker
                "hybrid_score": metadata.get("hybrid_score"),
                "retrieval_sources": metadata.get("retrieval_sources"),
                "raw_rerank_score": metadata.get("raw_rerank_score"),
                "rerank_boost": metadata.get("rerank_boost"),
                "rerank_boost_reasons": metadata.get("rerank_boost_reasons"),
                "rerank_score": metadata.get("rerank_score"),
            }

            sources.append(source)

        return sources

    def invoke(
            self,
            question: str,
            retrieve_top_k: int = 10,
            rerank_top_k: int = 5,
            metadata_filter: dict | None = None,
            enable_rewrite: bool = True,
            enable_rerank: bool = True,
    ) -> dict:
        """
        执行一次企业级 RAG 问答。

        流程：
        1. 保存用户原始问题。
        2. 可选：使用 Query Rewriter 改写问题，提高检索效果。
        3. 使用 Retriever 检索候选文档。
        4. 可选：使用 Reranker 对候选文档重排。
        5. 使用 Confidence Guard 判断检索结果是否可信。
        6. 如果低置信度，直接拒答，不进入 LLM。
        7. 如果置信度通过，把上下文交给 LLM 生成答案。
        8. 如果 LLM 最终拒答，则清空 sources，避免返回无关引用。
        """

        # 统一拒答文本。
        # 后面所有拒答都用同一句，方便 API 层、测试和前端统一判断。
        no_info_answer = "知识库中未找到相关信息。"

        # 1. 保存原始问题。
        # 最终回答时仍然使用 original_question，避免用户看到被改写后的问题。
        original_question = question

        # 2. 查询改写。
        # 检索用 rewritten_question，回答展示仍用 original_question。
        if enable_rewrite and self.rewriter is not None:
            rewritten_question = self.rewriter.rewrite(original_question)
        else:
            rewritten_question = original_question

        # 3. 初始检索。
        # metadata_filter 已经在 API 层根据用户角色做过权限过滤。
        retrieved_docs = self.retriever.retrieve(
            query=rewritten_question,
            top_k=retrieve_top_k,
            metadata_filter=metadata_filter,
        )

        # 4. Reranker 重排。
        # 如果关闭 rerank，直接取检索结果前 rerank_top_k 个。
        if enable_rerank and self.reranker is not None:
            final_docs = self.reranker.rerank(
                query=rewritten_question,
                documents=retrieved_docs,
                top_k=rerank_top_k,
            )
        else:
            final_docs = retrieved_docs[:rerank_top_k]

        # 5. 低置信度拒答。
        # 如果检索结果明显和问题无关，不进入 LLM，直接返回拒答。
        confidence_result = None

        if self.confidence_guard is not None:
            confidence_result = self.confidence_guard.check(
                query=rewritten_question,
                documents=final_docs,
            )

            if not confidence_result.passed:
                return {
                    "question": original_question,
                    "rewritten_question": rewritten_question,
                    "answer": no_info_answer,
                    "sources": [],
                    "retrieved_docs": final_docs,
                    "metadata_filter": metadata_filter,
                    "confidence": confidence_result.to_dict(),
                }

            # 临时调试：检查最终送给 LLM 的文档
            print("=" * 80)
            print("[DEBUG Final Docs Before LLM]")
            print("confidence passed:", confidence_result.passed)
            print("confidence reason:", confidence_result.reason)
            print("final docs count:", len(final_docs))

            for index, doc in enumerate(final_docs, start=1):
                print(
                    f"doc {index}:",
                    "source=", doc.metadata.get("source"),
                    "header_2=", doc.metadata.get("header_2"),
                )
                print(
                    "content:",
                    doc.page_content[:300].replace("\n", " "),
                )

            print("=" * 80)

        # 6. 构造上下文。
        # 这里不依赖 self.format_context()，直接在 invoke 内部拼接上下文，
        # 避免当前类中没有 format_context 方法导致 AttributeError。
        context_parts = []

        for index, doc in enumerate(final_docs, start=1):
            metadata = doc.metadata or {}

            source = metadata.get("source", "unknown")
            header_1 = metadata.get("header_1", "")
            header_2 = metadata.get("header_2", "")
            chunk_id = metadata.get("chunk_id", "")

            context_parts.append(
                f"【文档 {index}】\n"
                f"来源：{source}\n"
                f"一级标题：{header_1}\n"
                f"二级标题：{header_2}\n"
                f"切片ID：{chunk_id}\n"
                f"内容：\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # 7. 构造 Prompt。
        # 规则重点：
        # - 只能根据知识库上下文回答
        # - 不允许使用常识、猜测、外部知识
        # - 找不到信息时必须返回固定拒答文本
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
        你是企业知识库问答助手。

        请严格遵守以下规则：

        1. 只能根据【知识库上下文】回答，不得使用外部知识。
        2. 不得编造制度、金额、日期、审批人、权限范围或例外情况。
        3. 如果上下文能够通过明确规则回答用户问题，即使没有出现完全相同的问法，也应基于规则作出有限、可解释的判断。
        4. 允许进行以下类型的有限业务推理：
           - 将酒店星级问题转换为住宿费用是否超过标准；
           - 将口语问题对应到上下文中的正式制度术语；
           - 根据“标准、条件、审批规则”回答是否允许及需要什么手续。
        5. 有限推理必须能从上下文直接推出，不得补充上下文没有提供的事实。
        6. 如果上下文没有直接规定“可以”或“不可以”，但提供了判断条件，应明确说明：
           - 制度未直接按该维度限制；
           - 实际应依据哪些条件判断。
        7. 如果上下文只有部分信息，只回答能够被支持的部分。
        8. 只有在上下文完全没有可用于回答的信息时，才只回答：
           知识库中未找到相关信息。
        9. 涉及金额、时间、天数和审批流程时，必须严格依据上下文。
        10. 回答应简洁、清楚，优先直接回答用户问题。
        """,
                ),
                (
                    "human",
                    """
        【用户原始问题】
        {original_question}

        【改写后的检索问题】
        {rewritten_question}

        【知识库上下文】
        {context}

        请根据知识库上下文回答用户原始问题。
        """,
                ),
            ]
        )

        # 8. 调用 LLM。
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke(
            {
                "original_question": original_question,
                "rewritten_question": rewritten_question,
                "context": context,
            }
        )

        # 9. 清理答案前后空白。
        answer = answer.strip()

        # 10. sources 处理。
        # 如果最终答案是拒答，则 sources 必须为空。
        # 否则会出现“答案说没找到，但还返回了无关来源”的问题。
        if answer == no_info_answer:
            sources = []
        else:
            sources = self.format_sources(final_docs)

        # 11. 返回完整结果。
        return {
            "question": original_question,
            "rewritten_question": rewritten_question,
            "answer": answer,
            "sources": sources,
            "retrieved_docs": final_docs,
            "metadata_filter": metadata_filter,
            "confidence": confidence_result.to_dict() if confidence_result else None,
        }



